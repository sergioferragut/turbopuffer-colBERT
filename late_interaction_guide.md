# Late Interaction Search with ColBERT on turbopuffer

turbopuffer supports vector and BM25 full-text search. For retrieval tasks where neither alone is sufficient — long documents, rare terms, multi-concept queries — late interaction with ColBERT closes the gap.

---

## What Is Late Interaction?

Dense retrieval compresses each document into a single vector. A query becomes a single vector too, and ranking is a dot product. Fast, but lossy — a 500-word article gets squeezed into 768 numbers.

**Late interaction** keeps every token's embedding. A 100-token document produces 100 vectors (128-dim each with ColBERT). At query time, each query token finds its best match in the document's token pool. The document score is the sum of those per-token maximum similarities — the **MaxSim** operator.

```
Score(q, d) = Σ_{i ∈ query_tokens} max_{j ∈ doc_tokens} sim(q_i, d_j)
```

The analogy: dense retrieval summarizes a book into one sentence before you arrive. Late interaction lets you skim every sentence when you get there.

### When to use it

- Long documents with heterogeneous content (legal, medical, code)
- Queries with rare or specific terminology that dense models gloss over
- Multi-concept queries where different parts of the document answer different parts of the query
- Any domain where precision matters more than raw speed

### When not to use it

- Sub-50ms latency requirements — late interaction adds encoding and multiple query round trips
- Short, homogeneous documents where dense retrieval already performs well
- Cost-constrained workloads where 10–20× more storage is prohibitive

---

## How It Works on turbopuffer

turbopuffer stores one vector per row. Multi-vector documents map to multiple rows — one per token embedding. Each token row carries the document ID and token position as attributes.

```
┌─────────────────────────────────────────────────────────┐
│  Query: "what causes inflation"                         │
└──────────────────────┬──────────────────────────────────┘
                       │
              ColBERT encode
                       │
         ┌─────────────▼─────────────┐
         │  Q token vectors (Q × 128) │
         │  q_0, q_1, ..., q_14       │
         └──────────┬────────────────┘
                    │
         batch via multi-query API (16 vecs/call)
                    │
         ┌──────────▼────────────────┐
         │  turbopuffer ANN search    │
         │  top-50 token rows / query │
         │  vector → (doc_id, dist)   │
         └──────────┬────────────────┘
                    │
         collect candidate doc_ids
         accumulate per-(doc, q_token) max sim
                    │
         ┌──────────▼────────────────┐
         │  MaxSim rerank (client)    │
         │  score(d) = Σ max_sim(q_i) │
         └──────────┬────────────────┘
                    │
              top-k documents
```

**Storage layout.** Each token row in turbopuffer contains:

| Field | Type | Notes |
|-------|------|-------|
| `id` | uint | Sequential row ID |
| `vector` | float32[128] | Token embedding, L2-normalized |
| `doc_id` | string | Source document identifier |
| `token_pos` | uint | Position within document |

The `doc_id` attribute is filterable — you can scope searches to subsets of your corpus. `token_pos` is marked non-filterable for a 50% storage discount.

---

## Implementation

### A. Install

```bash
pip install turbopuffer datasets transformers torch numpy huggingface_hub safetensors
```

### B. ColBERT Encoder

ColBERT's architecture is BERT (768-dim) + linear projection (768 → 128) + L2 normalization.
`BertModel.from_pretrained` loads the BERT weights; we extract the linear layer separately.

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, BertModel
from huggingface_hub import snapshot_download

COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"
MAX_DOC_TOKENS = 180   # ColBERT default; truncate longer documents
MAX_QUERY_TOKENS = 32  # ColBERT pads/truncates queries to 32 tokens
DIM = 128              # ColBERT output dimension


class ColBERTEncoder:
    """
    Thin wrapper around ColBERTv2 for encoding text into token-level embeddings.

    Documents get a [D] prefix; queries get a [Q] prefix. This is part of
    ColBERT's training protocol and must be preserved at inference time.
    """

    def __init__(self, checkpoint: str = COLBERT_CHECKPOINT, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Load BERT backbone (ignores the linear.weight key it doesn't know)
        self.bert = BertModel.from_pretrained(checkpoint).to(device)
        self.bert.eval()

        # Load full checkpoint to extract ColBERT's linear projection weights.
        # The projection maps BERT's 768-dim CLS + token outputs → 128-dim.
        ckpt_dir = snapshot_download(checkpoint)
        safetensors = os.path.join(ckpt_dir, "model.safetensors")
        pytorch_bin = os.path.join(ckpt_dir, "pytorch_model.bin")

        if os.path.exists(safetensors):
            from safetensors.torch import load_file
            state = load_file(safetensors, device=device)
        else:
            state = torch.load(pytorch_bin, map_location=device)

        self.linear = nn.Linear(768, DIM, bias=False).to(device)
        self.linear.weight = nn.Parameter(state["linear.weight"])
        self.linear.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        doc: bool = True,
        max_length: int | None = None,
    ) -> list[np.ndarray]:
        """
        Encode a list of texts into ColBERT token embeddings.

        Returns a list of arrays, one per input text, each shaped
        (num_tokens, 128). Padding tokens are stripped.
        """
        if max_length is None:
            max_length = MAX_DOC_TOKENS if doc else MAX_QUERY_TOKENS

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

        # Project 768 → 128 and L2-normalize
        embs = self.linear(output.last_hidden_state)  # (batch, seq, 128)
        embs = F.normalize(embs, p=2, dim=-1)

        # Strip padding tokens; return one array per document
        results = []
        for emb, mask in zip(embs, tokens["attention_mask"]):
            n_tokens = int(mask.sum())
            results.append(emb[:n_tokens].cpu().numpy())

        return results
```

### C. Index Documents into turbopuffer

Each document produces one turbopuffer row per token embedding. Batch upserts
of ~1,000 rows keep throughput high and cost low.

```python
import turbopuffer as tpuf

# turbopuffer v1 client: create a client then access namespaces via .namespace()
client = tpuf.Turbopuffer(
    api_key=os.environ["TURBOPUFFER_API_KEY"],
    region=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
)
ns = client.namespace("late-interaction-quora")

# Schema defined once; turbopuffer infers types from first write if omitted,
# but explicit schema lets us control filterability and storage cost.
SCHEMA = {
    "doc_id": {"type": "string"},
    # token_pos is stored for debugging but never filtered — 50% cost discount
    "token_pos": {"type": "uint", "filterable": False},
}


def index_documents(
    encoder: ColBERTEncoder,
    documents: list[dict],  # [{"id": "doc_0", "text": "..."}, ...]
    encode_batch: int = 64,
    upsert_batch: int = 1000,
) -> int:
    """
    Encode documents and write one row per token embedding into turbopuffer.
    Returns total rows written.
    """
    row_id = 0
    pending: list[dict] = []

    for i in range(0, len(documents), encode_batch):
        doc_batch = documents[i : i + encode_batch]
        texts = [d["text"] for d in doc_batch]

        # Encode batch → list of (num_tokens, 128) arrays
        token_emb_lists = encoder.encode(texts, doc=True)

        for doc, token_embs in zip(doc_batch, token_emb_lists):
            for pos, emb in enumerate(token_embs):
                pending.append({
                    "id": row_id,
                    "vector": emb.tolist(),
                    "doc_id": doc["id"],
                    "token_pos": pos,
                })
                row_id += 1

            # Flush when the pending batch is large enough
            if len(pending) >= upsert_batch:
                _write(pending)
                pending.clear()
                print(f"  {row_id:,} rows written...")

    if pending:
        _write(pending)

    print(f"Done. {row_id:,} token rows for {len(documents):,} documents.")
    return row_id


def _write(rows: list[dict]) -> None:
    # ns.write() is the turbopuffer v1 SDK method for inserting/upserting rows
    ns.write(
        upsert_rows=rows,
        distance_metric="cosine_distance",
        schema=SCHEMA,
    )
```

Load the Quora dataset and index it:

```python
from datasets import load_dataset

# Load 5,000 question pairs from the Quora dataset
ds = load_dataset("quora", split="train[:5000]")

# Extract unique questions; each example has a pair
questions: dict[int, str] = {}
for example in ds:
    for qid, qtext in zip(
        example["questions"]["id"], example["questions"]["text"]
    ):
        questions[qid] = qtext

documents = [{"id": str(qid), "text": text} for qid, text in questions.items()]
doc_store = {str(qid): text for qid, text in questions.items()}

print(f"{len(documents):,} unique questions to index")

encoder = ColBERTEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
total_rows = index_documents(encoder, documents)
```

### D. Late Interaction Query

For each query token vector, turbopuffer's multi-query API batches up to 16 searches
into a single round trip. MaxSim is computed client-side from the returned distances.

```python
from collections import defaultdict


def late_interaction_search(
    query: str,
    doc_store: dict[str, str],
    encoder: ColBERTEncoder,
    top_k: int = 10,
    candidates_per_token: int = 100,
) -> list[dict]:
    """
    ColBERT-style late interaction search over turbopuffer.

    Returns top_k documents sorted by MaxSim score (descending).
    Each result: {"doc_id": str, "score": float, "text": str}
    """
    # Step 1: Encode the query into Q token vectors, each 128-dim
    query_embs = encoder.encode([query], doc=False)[0]  # shape: (Q, 128)

    # Step 2: Search turbopuffer for each query token vector.
    # multi_query fires up to 16 sub-queries atomically against the same
    # namespace snapshot in a single round trip.
    #
    # cosine_distance = 1 - cosine_similarity for L2-normalised vectors,
    # so similarity = 1 - row["$dist"].
    MULTI_QUERY_LIMIT = 16
    doc_token_sims: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for batch_start in range(0, len(query_embs), MULTI_QUERY_LIMIT):
        batch = query_embs[batch_start : batch_start + MULTI_QUERY_LIMIT]

        # ns.multi_query is the dedicated batch-query method in turbopuffer v1 SDK
        response = ns.multi_query(
            queries=[
                {
                    "rank_by": ("vector", "ANN", vec.tolist()),
                    "top_k": candidates_per_token,
                    "include_attributes": ["doc_id"],
                }
                for vec in batch
            ]
        )

        # response.results[i].rows is the result list for batch[i]
        for offset, result in enumerate(response.results):
            q_idx = batch_start + offset
            if result.rows is None:
                continue

            for row in result.rows:
                doc_id = row["doc_id"]          # extra attributes via row["attr"]
                sim = 1.0 - float(row["$dist"]) # cosine distance → similarity

                # Keep the maximum similarity this query token achieved
                # against any token in this document (MaxSim accumulator)
                if sim > doc_token_sims[doc_id][q_idx]:
                    doc_token_sims[doc_id][q_idx] = sim

    if not doc_token_sims:
        return []

    # Step 3: Aggregate MaxSim score per document.
    # MaxSim(q, d) = sum over query tokens of max-sim-to-any-doc-token.
    # doc_token_sims[doc_id][q_idx] already holds the max sim per (doc, q_token).
    maxsim_scores = {
        doc_id: sum(q_sims.values())
        for doc_id, q_sims in doc_token_sims.items()
    }

    # Step 4: Return top_k by MaxSim score
    ranked = sorted(maxsim_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "doc_id": doc_id,
            "score": round(score, 4),
            "text": doc_store.get(doc_id, ""),
        }
        for doc_id, score in ranked[:top_k]
    ]
```

Example usage:

```python
results = late_interaction_search(
    query="how to reduce government deficit spending",
    doc_store=doc_store,
    encoder=encoder,
    top_k=5,
)

for r in results:
    print(f"[{r['score']:.4f}] {r['text'][:80]}")
```

### E. Dense Baseline

For comparison: mean-pool ColBERT's BERT backbone to get a single 768-dim vector
per document. One vector per document, one turbopuffer query per search.

```python
# Separate namespace for the dense baseline
ns_dense = client.namespace("dense-baseline-quora")


@torch.no_grad()
def encode_dense(
    encoder: ColBERTEncoder,
    texts: list[str],
    max_length: int = 180,
) -> np.ndarray:
    """
    Mean-pool BERT's last hidden state (excluding padding) to get a single
    768-dim vector per text. Used for the dense retrieval baseline.
    """
    tokens = encoder.tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(encoder.device)

    output = encoder.bert(**tokens)

    # Masked mean pool: ignore padding tokens
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    summed = (output.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = F.normalize(summed / counts, p=2, dim=-1)

    return embeddings.cpu().numpy()


def index_dense(encoder: ColBERTEncoder, documents: list[dict], batch_size: int = 128):
    """Index one 768-dim vector per document."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        embs = encode_dense(encoder, [d["text"] for d in batch])

        # IDs must be integers or UUIDs; hash string IDs to uint64 if needed
        ns_dense.write(
            upsert_rows=[
                {"id": int(d["id"]) if d["id"].isdigit() else abs(hash(d["id"])) % (2**63),
                 "vector": emb.tolist(),
                 "doc_id": d["id"]}
                for d, emb in zip(batch, embs)
            ],
            distance_metric="cosine_distance",
        )
    print(f"Dense index: {len(documents):,} documents.")


def dense_search(
    query: str,
    doc_store: dict[str, str],
    encoder: ColBERTEncoder,
    top_k: int = 10,
) -> list[dict]:
    """Single-vector dense retrieval: one turbopuffer query per search."""
    emb = encode_dense(encoder, [query])[0]

    response = ns_dense.query(
        rank_by=("vector", "ANN", emb.tolist()),
        top_k=top_k,
        include_attributes=["doc_id"],
    )

    if response.rows is None:
        return []

    return [
        {
            "doc_id": row["doc_id"],
            "score": round(1.0 - float(row["$dist"]), 4),
            "text": doc_store.get(str(row["doc_id"]), ""),
        }
        for row in response.rows
    ]
```

---

## Analysis

Results measured on two datasets with very different document lengths. Hardware:
MacBook Pro M3, CPU inference, turbopuffer warm cache (gcp-us-central1).
50 randomly sampled pairs per evaluation set.

### Datasets

| Dataset | Corpus | Avg tokens/doc | Task |
|---------|--------|---------------|------|
| Quora | 9,859 unique questions | **18.8** | Find the duplicate question |
| SQuAD v1.1 | 1,867 Wikipedia passages | **129.7** | Find the source passage for a question |

### Result Quality (Recall@10)

For each evaluation query, we check whether the known relevant document appears
in the top-10 results.

| Dataset | Method | Recall@10 | Hits |
|---------|--------|-----------|------|
| Quora | Dense (BERT mean-pool, 768-dim) | 0.940 | 47/50 |
| Quora | Late Interaction (ColBERT MaxSim) | **0.960** | 48/50 |
| SQuAD | Dense (BERT mean-pool, 768-dim) | 0.880 | 44/50 |
| SQuAD | Late Interaction (ColBERT MaxSim) | **0.940** | 47/50 |

The advantage of late interaction scales with document length. On Quora's short
questions (18.8 tokens), the recall gap is **+0.020**. On SQuAD's Wikipedia passages
(129.7 tokens), the gap widens to **+0.060** — 3× larger.

This confirms the core hypothesis: longer, information-dense documents benefit more
from token-level alignment. Dense retrieval compresses a 130-token passage into one
vector; late interaction can match individual query terms against individual passage
tokens.

**Where late interaction wins on SQuAD:**

| Query | Dense top-1 | Late Interaction result |
|-------|-------------|------------------------|
| "where did the trade route pass through?" | Unrelated Congo passage | ✓ Correct passage about the Yongle road |
| "What was the name of the free music promotion on Kanye's website in 2010?" | Unrelated Apple iTunes article | ✓ Correct Kanye West album passage |
| "Beyonce's younger sibling also sang with her in what band?" | Wrong Beyoncé passage (about Michael Jackson influence) | ✓ Correct passage about Destiny's Child |

In each SQuAD miss, dense retrieval found a semantically related passage (same
topic area) but not the specific one containing the answer. Late interaction's
token alignment pinpoints the passage with the matching named entities and events.

**Where late interaction wins on Quora:**

The Quora wins involve **rephrased questions** where individual token overlap
(GATE→CAT, "needing improvement"→"need to improve") matters more than overall
semantic similarity. Dense retrieval glosses over lexical variation; late interaction
preserves it.

### Latency

Measured over 50 queries, warm turbopuffer cache.

| Dataset | Method | Avg (ms) | P95 (ms) | tpuf calls/query |
|---------|--------|----------|----------|-----------------|
| Quora | Dense | 128.8 | 149.6 | 1 |
| Quora | Late Interaction | 251.6 | 373.5 | 2 (batched) |
| SQuAD | Dense | 126.9 | 145.2 | 1 |
| SQuAD | Late Interaction | 278.5 | 368.1 | 2 (batched) |

Dense latency is stable at ~128ms regardless of document length — it is always one
single-vector query. Late interaction is **2–2.2× slower** on both datasets, and
SQuAD's overhead is marginally higher because longer documents produce more candidate
rows to scan. The `multi_query` batching caps late interaction at 2 round trips for
queries up to 32 tokens (the ColBERT default).

On GPU inference, the encoding step drops from ~80ms to <5ms, reducing total
late interaction latency to ~170ms and closing the gap to ~1.3×.

### Cost

Storage is billed on logical bytes. f32 vectors: 4 bytes/dimension.
Non-filterable attributes receive a 50% storage discount.

**Storage per 100K documents:**

| Dataset | Method | Avg tokens/doc | Rows | Vector dims | Relative storage |
|---------|--------|---------------|------|-------------|-----------------|
| Quora | Dense | 18.8 | 100K | 768 | 1× |
| Quora | Late Interaction | 18.8 | 1.88M | 128 | **3.1×** |
| SQuAD | Dense | 129.7 | 100K | 768 | 1× |
| SQuAD | Late Interaction | 129.7 | 12.97M | 128 | **21.6×** |

The storage multiplier scales with tokens/doc. For Quora (18.8 tokens), the 6×
dimension reduction (768→128) almost cancels the row explosion (18.8×), giving
only 3.1× overhead. For SQuAD (129.7 tokens), the row explosion dominates: 21.6×.

The formula: `storage_ratio = avg_tokens_per_doc × (128 / 768)`.
At ~47 tokens/doc the ratio reaches 8×; beyond that, late interaction storage cost
grows linearly with document length.

**Storage savings with f16:** Adding `"vector": {"type": "f16"}` to your schema
halves vector storage with negligible accuracy impact at 128-dim. This brings
SQuAD-scale storage overhead to ~10.8× vs dense.

**Query cost:**

Late interaction costs more per query because it scans a larger namespace.
For short documents (Quora), this is ~6× more than dense. For longer documents
(SQuAD), the larger namespace makes late interaction ~20× more expensive per query.

**Tradeoff summary:**

| Doc length | Recall gain | Storage overhead | Query overhead | Verdict |
|-----------|------------|-----------------|---------------|---------|
| Short (< 50 tokens) | +2pp | 3× | 6× | Consider dense + BM25 hybrid first |
| Medium (50–150 tokens) | +3–5pp | 8–21× | 10–20× | Late interaction earns its cost |
| Long (> 150 tokens) | +5pp+ | > 21× | > 20× | Late interaction is often the right choice |

For short-document corpora where dense retrieval already scores >0.92 Recall@10,
consider dense + BM25 hybrid (see [hybrid search guide](https://turbopuffer.com/docs/hybrid-search))
as a cheaper alternative. Late interaction's advantage compounds on longer,
heterogeneous documents where token-level alignment is the only way to match
specific terminology across passage boundaries.

---

## Tips

**Token budget.** ColBERT truncates at 180 tokens by default. For longer documents,
chunk by paragraph and store a `chunk_id` attribute. Include the parent document ID
separately so you can group results.

```python
# Long-document chunking
import textwrap

def chunk_document(doc_id: str, text: str, chunk_size: int = 180) -> list[dict]:
    words = text.split()
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [
        {"id": f"{doc_id}_c{i}", "doc_id": doc_id, "chunk_id": i,
         "text": " ".join(chunk)}
        for i, chunk in enumerate(chunks)
    ]
```

After reranking, deduplicate by `doc_id` and keep the highest-scoring chunk per document.

**Batch your upserts.** Rows-at-a-time upserts are slow and expensive. Always batch
to at least 512 rows; 1,000–5,000 is typical. The API accepts up to 512 MB per request.

**Use filters to narrow candidates.** If your corpus has natural partitions (user ID,
language, date range), add a filterable attribute and push the filter into each ANN
sub-query. This reduces the scanned namespace size and improves both latency and cost.

```python
# Filter to documents from a specific source before ANN
{
    "rank_by": ("vector", "ANN", vec.tolist()),
    "top_k": 100,
    "filters": ("source", "Eq", "legal"),
    "include_attributes": ["doc_id"],
}
```

**Warm the cache.** turbopuffer's first query against a cold namespace hits object
storage. Issue a dummy query at startup:

```python
# Warm-up: run once before serving real traffic
_ = ns.query(rank_by=("vector", "ANN", [0.0] * DIM), top_k=1)
```

**Use f16 vectors to halve storage.** Declare the namespace schema with `"vector": {"type": "f16"}` to store token embeddings in half precision. For L2-normalized ColBERT embeddings at 128-dim, f16 has negligible accuracy impact.

```python
SCHEMA_F16 = {
    "doc_id": {"type": "string"},
    "token_pos": {"type": "uint", "filterable": False},
    "vector": {"type": "f16"},
}
```

Note: vectors are stored as f16 but written as f32 (the API upsamples on write).
