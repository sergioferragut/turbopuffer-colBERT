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

UPSERT_BATCH = 1000   # rows per write call
ENCODE_BATCH = 64     # documents per encoding batch


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
    Returns total token rows written.
    """
    ns = client.namespace(cfg.ns_late)
    documents = cfg.documents

    row_id = 0
    pending: list[dict] = []

    for i in range(0, len(documents), ENCODE_BATCH):
        doc_batch = documents[i : i + ENCODE_BATCH]
        token_emb_lists = encoder.encode([d["text"] for d in doc_batch], doc=True)

        for doc, token_embs in zip(doc_batch, token_emb_lists):
            for pos, emb in enumerate(token_embs):
                pending.append({
                    "id": row_id,
                    "vector": emb.tolist(),
                    "doc_id": doc["id"],
                    "token_pos": pos,
                })
                row_id += 1

            if len(pending) >= UPSERT_BATCH:
                ns.write(upsert_rows=pending, distance_metric="cosine_distance", schema=SCHEMA_LATE)
                pending.clear()
                print(f"  {row_id:,} token rows ({i + ENCODE_BATCH}/{len(documents)} docs)...")

    if pending:
        ns.write(upsert_rows=pending, distance_metric="cosine_distance", schema=SCHEMA_LATE)

    print(f"Late interaction: {row_id:,} token rows for {len(documents):,} docs "
          f"(avg {row_id/len(documents):.1f} tokens/doc)")
    return row_id


# ── dense indexing ─────────────────────────────────────────────────────────


def index_dense(encoder: ColBERTEncoder, cfg: DatasetConfig) -> None:
    """Encode documents with mean-pooled BERT and write one 768-dim row per document."""
    ns = client.namespace(cfg.ns_dense)
    documents = cfg.documents

    for i in range(0, len(documents), ENCODE_BATCH):
        batch = documents[i : i + ENCODE_BATCH]
        embs = encoder.encode_dense([d["text"] for d in batch])

        rows = []
        for doc, emb in zip(batch, embs):
            numeric_id = int(doc["id"]) if doc["id"].isdigit() else abs(hash(doc["id"])) % (2**63)
            rows.append({"id": numeric_id, "vector": emb.tolist(), "doc_id": doc["id"]})

        ns.write(upsert_rows=rows, distance_metric="cosine_distance", schema=SCHEMA_DENSE)

        if (i // ENCODE_BATCH) % 10 == 0:
            print(f"  {min(i + ENCODE_BATCH, len(documents))}/{len(documents)} dense docs...")

    print(f"Dense: {len(documents):,} documents.")


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
