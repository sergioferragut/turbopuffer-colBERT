"""
ColBERT token-level encoder.

Architecture: BERT backbone (768-dim) + linear projection (768 → 128) + L2 norm.
BertModel.from_pretrained loads BERT weights; the linear layer is extracted
separately because BertModel doesn't know about the ColBERT-specific projection.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, BertModel

COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"
MAX_DOC_TOKENS = 180    # ColBERT default; truncate longer documents
MAX_QUERY_TOKENS = 32   # ColBERT pads/truncates queries to 32 tokens
DIM = 128               # ColBERT output dimension
BERT_DIM = 768          # BERT hidden size


class ColBERTEncoder:
    """
    Thin wrapper around ColBERTv2 for encoding text into token-level embeddings.

    Documents get a [D] prefix; queries get a [Q] prefix. This is part of
    ColBERT's training protocol and must be preserved at inference time.

    Usage:
        encoder = ColBERTEncoder()
        doc_embs = encoder.encode(["some document text"], doc=True)
        # doc_embs[0].shape == (num_tokens, 128)
    """

    def __init__(
        self,
        checkpoint: str = COLBERT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading ColBERT encoder from {checkpoint} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # BertModel.from_pretrained loads all matching BERT weights and silently
        # ignores unrecognised keys like "linear.weight".
        self.bert = BertModel.from_pretrained(checkpoint).to(device)
        self.bert.eval()

        # Extract ColBERT's linear projection from the full checkpoint.
        # This maps BERT's 768-dim token outputs → 128-dim embeddings.
        ckpt_dir = snapshot_download(checkpoint)
        state = _load_state_dict(ckpt_dir, device)

        self.linear = nn.Linear(BERT_DIM, DIM, bias=False).to(device)
        self.linear.weight = nn.Parameter(state["linear.weight"])
        self.linear.eval()
        print("Encoder ready.")

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        doc: bool = True,
        max_length: int | None = None,
    ) -> list[np.ndarray]:
        """
        Encode a list of texts into ColBERT token embeddings.

        Returns a list of float32 arrays, one per input text, each shaped
        (num_tokens, 128). Padding tokens are stripped from each array.

        Args:
            texts:      List of input strings.
            doc:        True for documents ([D] prefix), False for queries ([Q] prefix).
            max_length: Token budget. Defaults to MAX_DOC_TOKENS / MAX_QUERY_TOKENS.
        """
        if max_length is None:
            max_length = MAX_DOC_TOKENS if doc else MAX_QUERY_TOKENS

        # ColBERT requires [D] / [Q] prefixes to activate the correct attention
        # patterns learned during fine-tuning.
        marker = "[D]" if doc else "[Q]"
        marked = [f"{marker} {t}" for t in texts]

        tokens = self.tokenizer(
            marked,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.bert(**tokens)

        # Project 768 → 128 then L2-normalise so cosine similarity = dot product.
        embs = self.linear(output.last_hidden_state)  # (batch, seq, 128)
        embs = F.normalize(embs, p=2, dim=-1)

        # Strip padding; return one array per text.
        results: list[np.ndarray] = []
        for emb, mask in zip(embs, tokens["attention_mask"]):
            n_tokens = int(mask.sum())
            results.append(emb[:n_tokens].cpu().numpy().astype(np.float32))

        return results

    @torch.no_grad()
    def encode_dense(
        self,
        texts: list[str],
        max_length: int = MAX_DOC_TOKENS,
    ) -> np.ndarray:
        """
        Mean-pool BERT's last hidden state (excluding padding) to produce a single
        768-dim vector per text. Used for the dense retrieval baseline.

        Returns float32 array of shape (len(texts), 768).
        """
        tokens = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.bert(**tokens)

        # Masked mean pool: sum then divide by non-padding token count.
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = F.normalize(summed / counts, p=2, dim=-1)

        return embeddings.cpu().numpy().astype(np.float32)


# ── helpers ────────────────────────────────────────────────────────────────


def _load_state_dict(ckpt_dir: str, device: str) -> dict:
    """Load model weights from safetensors or pytorch_model.bin."""
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    pytorch_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path, device=device)

    if os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location=device, weights_only=True)

    raise FileNotFoundError(
        f"No model weights found in {ckpt_dir}. "
        "Expected model.safetensors or pytorch_model.bin."
    )
