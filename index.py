"""
Index a dataset into turbopuffer for late-interaction and dense retrieval.

Creates two namespaces per dataset:
  late-interaction-{dataset} : one 128-dim row per token embedding (ColBERT)
  dense-baseline-{dataset}   : one 768-dim row per document (mean-pool BERT)

Usage:
    python index.py --dataset quora   # default
    python index.py --dataset squad
"""

import argparse

from config import client
from dataset_loaders import DatasetConfig, load
from encoder import ColBERTEncoder

UPSERT_BATCH = 10_000  # rows per write call
ENCODE_BATCH = 64     # documents per encoding batch
CHUNK_WORDS = 120     # target words per chunk — ~150 tokens, safely under MAX_DOC_TOKENS=180
CHUNK_OVERLAP = 20    # overlapping words between consecutive chunks


# ── chunking ───────────────────────────────────────────────────────────────


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Documents shorter than CHUNK_WORDS are returned as-is (single chunk).
    Longer documents are split with CHUNK_OVERLAP words of overlap between
    consecutive chunks so context at boundaries is not lost.
    """
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]

    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


# ── schema ─────────────────────────────────────────────────────────────────

SCHEMA_LATE = {
    "doc_id": {"type": "string"},
    # token_pos stored for debugging; non-filterable = 50% storage discount
    "token_pos": {"type": "uint", "filterable": False},
}

SCHEMA_DENSE = {
    "doc_id": {"type": "string"},
}


# ── late-interaction indexing ──────────────────────────────────────────────


def index_late_interaction(
    encoder: ColBERTEncoder,
    cfg: DatasetConfig,
) -> int:
    """
    Encode documents with ColBERT and write one row per token embedding.

    Long documents are split into overlapping chunks before encoding. All
    chunks of a document share the same doc_id, so MaxSim at query time
    naturally aggregates across all chunks.

    Returns total token rows written.
    """
    ns = client.namespace(cfg.ns_late)
    if ns.exists():
        ns.delete_all()

    # Expand documents into (doc_id, chunk_text) pairs.
    chunks: list[tuple[str, str]] = []
    for doc in cfg.documents:
        for chunk in _chunk_text(doc["text"]):
            chunks.append((doc["id"], chunk))

    n_docs = len(cfg.documents)
    n_chunks = len(chunks)
    if n_chunks > n_docs:
        print(f"  Chunked {n_docs:,} docs into {n_chunks:,} chunks "
              f"(avg {n_chunks/n_docs:.1f} chunks/doc)")

    row_id = 0
    pending: list[dict] = []

    for i in range(0, len(chunks), ENCODE_BATCH):
        chunk_batch = chunks[i : i + ENCODE_BATCH]
        token_emb_lists = encoder.encode([text for _, text in chunk_batch], doc=True)

        for (doc_id, _), token_embs in zip(chunk_batch, token_emb_lists):
            for pos, emb in enumerate(token_embs):
                pending.append({
                    "id": row_id,
                    "vector": emb.tolist(),
                    "doc_id": doc_id,
                    "token_pos": pos,
                })
                row_id += 1

            if len(pending) >= UPSERT_BATCH:
                ns.write(upsert_rows=pending, distance_metric="cosine_distance", schema=SCHEMA_LATE)
                pending.clear()
                print(f"  {row_id:,} token rows ({i + ENCODE_BATCH}/{n_chunks} chunks)...")

    if pending:
        ns.write(upsert_rows=pending, distance_metric="cosine_distance", schema=SCHEMA_LATE)

    print(f"Late interaction: {row_id:,} token rows for {n_docs:,} docs / {n_chunks:,} chunks "
          f"(avg {row_id/n_docs:.1f} tokens/doc)")
    return row_id


# ── dense indexing ─────────────────────────────────────────────────────────


def index_dense(encoder: ColBERTEncoder, cfg: DatasetConfig) -> None:
    """
    Encode documents with mean-pooled BERT and write one 768-dim row per chunk.

    Long documents are split into overlapping chunks. Each chunk row carries
    the parent doc_id so dense_search can deduplicate back to document level.
    """
    ns = client.namespace(cfg.ns_dense)
    if ns.exists():
        ns.delete_all()

    # Expand documents into (doc_id, chunk_idx, chunk_text) triples.
    chunks: list[tuple[str, int, str]] = []
    for doc in cfg.documents:
        for idx, chunk in enumerate(_chunk_text(doc["text"])):
            chunks.append((doc["id"], idx, chunk))

    for i in range(0, len(chunks), ENCODE_BATCH):
        batch = chunks[i : i + ENCODE_BATCH]
        embs = encoder.encode_dense([text for _, _, text in batch])

        rows = []
        for (doc_id, chunk_idx, _), emb in zip(batch, embs):
            # Unique integer ID per chunk derived from doc_id + chunk index.
            chunk_key = f"{doc_id}_c{chunk_idx}"
            numeric_id = abs(hash(chunk_key)) % (2**63)
            rows.append({"id": numeric_id, "vector": emb.tolist(), "doc_id": doc_id})

        ns.write(upsert_rows=rows, distance_metric="cosine_distance", schema=SCHEMA_DENSE)

        if (i // ENCODE_BATCH) % 10 == 0:
            print(f"  {min(i + ENCODE_BATCH, len(chunks))}/{len(chunks)} dense chunks...")

    print(f"Dense: {len(cfg.documents):,} documents / {len(chunks):,} chunks.")


# ── entrypoint ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quora", choices=["quora", "squad"],
                        help="Dataset to index (default: quora)")
    args = parser.parse_args()

    cfg = load(args.dataset)
    encoder = ColBERTEncoder()

    print(f"\n=== Indexing late-interaction namespace: {cfg.ns_late} ===")
    total_rows = index_late_interaction(encoder, cfg)

    print(f"\n=== Indexing dense namespace: {cfg.ns_dense} ===")
    index_dense(encoder, cfg)

    print(f"\nDone — {args.dataset}")
    print(f"  {cfg.ns_late} : {total_rows:,} token rows")
    print(f"  {cfg.ns_dense}   : {len(cfg.documents):,} document rows")
