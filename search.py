"""
Search functions for late interaction (ColBERT MaxSim) and dense retrieval.

Both functions return:
    [{"doc_id": str, "score": float, "text": str}, ...]
sorted descending by relevance score.

Usage:
    from encoder import ColBERTEncoder
    from search import late_interaction_search, dense_search

    encoder = ColBERTEncoder()
    doc_store = {"123": "What is inflation?", ...}

    results = late_interaction_search("causes of inflation", doc_store, encoder)
    for r in results:
        print(f"[{r['score']:.4f}] {r['text']}")
"""

from collections import defaultdict

from config import client
from encoder import ColBERTEncoder

# turbopuffer's multi_query batches up to this many vectors per round trip.
# ColBERT queries produce up to MAX_QUERY_TOKENS=32 vectors, so ceil(32/16)=2 calls.
MULTI_QUERY_BATCH = 16


# ── late interaction search ────────────────────────────────────────────────


def late_interaction_search(
    query: str,
    doc_store: dict[str, str],
    encoder: ColBERTEncoder,
    top_k: int = 10,
    candidates_per_token: int = 100,
    ns=None,
) -> list[dict]:
    """
    ColBERT-style late interaction search over turbopuffer.

    Algorithm:
      1. Encode query → Q token vectors (Q × 128), each L2-normalised.
      2. For each query token, ANN-search turbopuffer for the nearest token rows.
         Batched via multi_query (16 vectors per round trip).
      3. Accumulate per-(document, query-token) maximum similarity from results.
         Each result row carries a cosine distance; sim = 1 - dist.
      4. MaxSim score per document = sum over query tokens of that token's max sim.
      5. Return top_k documents sorted by MaxSim score.

    Args:
        query:               Query string.
        doc_store:           id → text mapping for result hydration.
        encoder:             Loaded ColBERTEncoder.
        top_k:               Number of results to return.
        candidates_per_token: How many nearest token rows to fetch per query token.
                              Higher = better recall, more compute. 100 is a good default.

    Returns:
        List of dicts: {"doc_id": str, "score": float, "text": str}
    """
    if ns is None:
        ns = client.namespace("late-interaction-quora")

    # Step 1: Encode the query into token vectors.
    query_embs = encoder.encode([query], doc=False)[0]  # shape: (Q, 128)

    # Step 2: ANN search for each query token vector.
    # doc_token_sims[doc_id][q_idx] = best similarity seen for this (doc, query-token) pair.
    # defaultdict initialises to 0.0, so missing pairs contribute 0 to MaxSim.
    doc_token_sims: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for batch_start in range(0, len(query_embs), MULTI_QUERY_BATCH):
        batch = query_embs[batch_start : batch_start + MULTI_QUERY_BATCH]

        # multi_query fires all sub-queries against the same namespace snapshot
        # in a single round trip.
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
                doc_id: str = row["doc_id"]
                # cosine_distance = 1 - cosine_similarity for L2-normalised vectors
                sim = 1.0 - float(row["$dist"])

                # Keep the best similarity this query token achieved for this doc.
                if sim > doc_token_sims[doc_id][q_idx]:
                    doc_token_sims[doc_id][q_idx] = sim

    if not doc_token_sims:
        return []

    # Step 3 & 4: MaxSim = sum over query tokens of per-token max similarity.
    maxsim_scores = {
        doc_id: sum(q_sims.values())
        for doc_id, q_sims in doc_token_sims.items()
    }

    # Step 5: Sort and return top_k.
    ranked = sorted(maxsim_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "doc_id": doc_id,
            "score": round(score, 4),
            "text": doc_store.get(doc_id, ""),
        }
        for doc_id, score in ranked[:top_k]
    ]


# ── dense search ───────────────────────────────────────────────────────────


def dense_search(
    query: str,
    doc_store: dict[str, str],
    encoder: ColBERTEncoder,
    top_k: int = 10,
    ns=None,
) -> list[dict]:
    """
    Single-vector dense retrieval: mean-pool BERT → one turbopuffer ANN query.

    Used as a baseline to compare against late interaction. Same BERT backbone,
    different aggregation: one vector per document instead of one per token.

    Args:
        query:    Query string.
        doc_store: id → text mapping for result hydration.
        encoder:  Loaded ColBERTEncoder.
        top_k:    Number of results to return.

    Returns:
        List of dicts: {"doc_id": str, "score": float, "text": str}
    """
    if ns is None:
        ns = client.namespace("dense-baseline-quora")

    # Encode query as a single 768-dim mean-pooled vector.
    emb = encoder.encode_dense([query])[0]  # shape: (768,)

    # Over-fetch to account for multiple chunks per document — after
    # deduplication we need at least top_k unique doc_ids remaining.
    response = ns.query(
        rank_by=("vector", "ANN", emb.tolist()),
        top_k=top_k * 5,
        include_attributes=["doc_id"],
    )

    if response.rows is None:
        return []

    # Deduplicate by doc_id, keeping the highest-scoring chunk per document.
    best: dict[str, float] = {}
    for row in response.rows:
        doc_id = str(row["doc_id"])
        score = 1.0 - float(row["$dist"])
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "doc_id": doc_id,
            "score": round(score, 4),
            "text": doc_store.get(doc_id, ""),
        }
        for doc_id, score in ranked[:top_k]
    ]

