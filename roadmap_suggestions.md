# Late Interaction: From Workaround to First-Class Feature

Internal proposal — turbopuffer engineering

---

## Executive Summary

Late interaction models (ColBERT, ColPali, ColQwen2) are gaining serious traction.
ColBERT is the go-to for text; ColPali is landing in multimodal search pipelines;
enterprise customers in legal and biomedical are asking about this pattern by name.
The current turbopuffer workaround — one row per token, client-side MaxSim, up to 2
API calls per query via `multi_query` (capped at 16 sub-queries) — works, but it
ships friction to the customer: non-obvious schema design, bespoke reranking code,
and latency that scales with query token count above the 16-query limit.
This proposal outlines five concrete changes, from a one-sprint API addition to a
longer-term architectural investment, that would make turbopuffer the obvious home
for late interaction workloads.

---

## Current State & Pain Points

A customer implementing late interaction today must:

1. **Issue multiple queries per search** — one per query token vector. A typical
   ColBERT query produces 15–32 token embeddings. `ns.multi_query()` batches up to
   16 sub-queries per round trip, so a 32-token query still requires 2 API calls.
   Queries under 16 tokens fit in 1 call; longer queries need client-side batching
   loops. Latency scales with query length beyond the 16-query cap.

2. **Implement MaxSim themselves** — after collecting ANN results, every customer
   writes their own aggregation loop: group by doc_id, take max sim per query token,
   sum across tokens. This logic is subtle (distance vs similarity sign, L2 norm
   assumption), implementation varies across teams, and bugs surface as silent
   ranking degradation.

3. **Manage the doc_id → token_id mapping** — there's no concept of a multi-vector
   document in turbopuffer. Customers invent their own naming schemes for token row
   IDs, write their own deduplication logic, and handle document deletion by tracking
   every token row ID they ever wrote. It's a footgun.

4. **Estimate storage costs without tooling** — a customer moving from dense to late
   interaction faces a step change in namespace size (typically 8–15×). There's no
   cost estimator that accounts for average tokens-per-document. They find out at
   billing time.

5. **Receive top-K token rows, not top-K documents** — turbopuffer's query API is
   vector-first. Returning the top-10 *documents* natively is not possible today;
   customers must over-fetch token rows and deduplicate. `limit.per` exists as a
   diversification primitive (`{"total": N, "per": {"attributes": ["doc_id"], "limit": 1}}`)
   but returns a 400 error when used with `rank_by=("vector", "ANN", ...)` — it is
   only supported on filter/scan queries. Customers are left to over-fetch and
   deduplicate in client code.

---

## Proposed API Changes

### Proposal 1: Batch Vector Query (Multi-Vector ANN)

**Problem:** ColBERT queries produce up to 32 token vectors. `ns.multi_query()` is
capped at 16 sub-queries per call, requiring 2 round trips for long queries and
a client-side batching loop.

**Current API (already working, just capped at 16):**

```python
# Today: up to 2 round trips for ColBERT (ceil(32/16) = 2)
for batch_start in range(0, len(query_token_vectors), 16):
    batch = query_token_vectors[batch_start : batch_start + 16]
    response = ns.multi_query(
        queries=[
            {
                "rank_by": ("vector", "ANN", vec.tolist()),
                "top_k": 50,
                "include_attributes": ["doc_id"],
            }
            for vec in batch
        ]
    )

# Proposed: 1 round trip, limit raised to 64
response = ns.multi_query(
    queries=[
        {
            "rank_by": ("vector", "ANN", vec.tolist()),
            "top_k": 50,
            "include_attributes": ["doc_id"],
        }
        for vec in query_token_vectors   # up to 32 vectors, fits in 1 call
    ]
)
# response.results[i] → sub-results for query_token_vectors[i]
```

The multi-query API already exists. Raising the limit from 16 to 64 (or removing
it for same-namespace queries) eliminates the client-side batching loop and the
second round trip entirely.

**Implementation complexity:** Low. The `queries` parameter already exists. This
is a limit increase and documentation improvement, not a new feature.

---

### Proposal 2: Native Multi-Vector Document Support

**Problem:** Customers manage token-row ↔ document mapping, document deletion,
and top-K document extraction themselves. No first-class document concept exists.

**Proposed API:**

```python
# Write: one logical document with N token vectors
ns.write(
    upsert_rows=[
        {
            "id": "doc_001",                               # document-level ID
            "multi_vector": [                              # (num_tokens, dim) matrix
                [0.12, -0.34, ..., 0.87],                 # token 0
                [0.05,  0.91, ..., -0.22],                # token 1
                # ...
            ],
            "source": "legal",                            # regular attributes
            "text": "The agreement shall remain in force...",
        }
    ],
    distance_metric="cosine_distance",
)

# Query: MaxSim aggregation server-side, returns top-K documents
results = ns.query(
    multi_vector=query_token_vectors,   # shape: (Q, dim)
    aggregation="maxsim",
    top_k=10,
    include_attributes=["text", "source"],
)
# results → top 10 documents, each with a maxsim score
```

Document deletion becomes `ns.delete(ids=["doc_001"])` — one call, all token
rows removed. No client-side bookkeeping.

**Implementation complexity:** High. Requires a new storage model (one document →
N vectors, shared ID), new ANN index structures (or sharding by document), and
server-side MaxSim aggregation at query time. This is the highest-leverage long-term
investment, but it's a significant architectural change.

**Suggested approach:** Start with a storage-layer abstraction that groups token
rows under a document ID without changing the ANN index. MaxSim aggregation runs
post-ANN, which is fast (linear scan over a small candidate set). This ships a
correct v1 without index surgery.

---

### Proposal 3: Server-Side MaxSim Reranking

**Problem:** Every customer implements MaxSim differently. Bugs are silent (wrong
sign, missing normalization) and degrade ranking quality without error.

**Proposed API:**

Extend `ns.multi_query()` with two new parameters: `rerank` and `top_k_after_rerank`.
The query token vectors are already in the request — the server unions candidates
across all sub-queries, applies MaxSim, and returns final top-K documents. No
additional input required from the caller.

```python
response = ns.multi_query(
    queries=[
        {
            "rank_by": ("vector", "ANN", vec.tolist()),
            "top_k": 100,                            # candidate pool per token
            "include_attributes": ["doc_id"],
        }
        for vec in query_token_vectors               # the full token matrix
    ],
    rerank={
        "method": "maxsim",
        "group_by": "doc_id",                        # union candidates, score per doc
    },
    top_k_after_rerank=10,                           # return final top-10 documents
)
# response.results → top-10 documents ranked by MaxSim score
```

The server already has everything it needs: the full set of query token vectors
(in the sub-queries), the ANN candidates with their `doc_id` attributes, and the
distances. MaxSim is just: for each `doc_id`, for each query token, take the
minimum distance seen across all token rows for that doc, then sum the converted
similarities. No mean-vector computation, no extra round trip, no client aggregation.

Storage layout stays the same as today. The reranking step is a union of candidate
sets (one per sub-query) followed by a dot-product loop — both cheap operations over
a small in-memory set.

**Implementation complexity:** Medium. The ANN passes are standard. The reranking
step is a post-processing aggregation in the query handler. The main design question
is the `group_by` attribute name (must be `doc_id` or a configurable field) and
whether the server re-fetches token vectors for candidate docs or uses the distances
already returned by the ANN pass (distances are sufficient for MaxSim; no re-fetch needed).

This proposal ships faster than Proposal 2 and delivers the highest immediate
customer value: correct, consistent MaxSim without any client code.

---

### Proposal 4: Dimension-Optimized Namespaces (128-dim)

**Problem:** ColBERT uses 128-dim vectors; most dense models use 768 or 1536.
Today all namespaces use the same storage path. Late interaction namespaces pay a
per-byte penalty that could be avoided with a namespace-level dim declaration.

**Proposed API:**

```python
# Declare small-dim namespace at creation time
ns = client.namespace("late-interaction-quora")
ns.write(
    upsert_rows=[...],
    schema={"vector": {"type": "f16", "dimensions": 128}},
    distance_metric="cosine_distance",
)
```

`"dimensions": 128` is already inferred from the first upsert. The ask is to make
this explicit in the namespace config so the storage layer can optimize block sizes,
cache allocation, and SIMD batch widths for 128-dim vectors specifically.

128-dim f16 = 256 bytes/vector vs 768-dim f32 = 3,072 bytes/vector. Late interaction
namespaces already have 10× more rows; right-sizing the storage layout reduces the
physical overhead per row and improves cache hit rates.

**Implementation complexity:** Low. This is a storage hint, not a new query
primitive. Primarily benefits the cache and ANN index packing; no API surface change
beyond the schema declaration.

---

### Proposal 5: Multi-Vector Cost Estimator in Docs and Dashboard

**Problem:** Customers don't realize their storage costs can be 3–22× higher for
late interaction until they've already indexed. The multiplier scales linearly with
average tokens per document: `avg_tokens × (128/768)`. Short-document corpora
(~19 tokens/doc) see ~3× overhead; long-document corpora (~130 tokens/doc) see ~22×.
This creates billing surprises and support tickets.

**Proposed addition to docs:**

```
Storage estimate for late interaction:

  Documents:       100,000
  Avg tokens/doc:  60          ← user fills this in
  Vector dim:      128
  Precision:       f16 (2 bytes/dim)

  Token rows:    100,000 × 60  = 6,000,000
  Vector bytes:  6M × 128 × 2  = 1.54 GB
  Attribute bytes (doc_id, token_pos): ~72 MB

  Estimated storage: ~1.6 GB/month
  Storage vs. dense baseline: ~10× (formula: avg_tokens × 128/768)
  At current rate:   $X.XX/month  ← link to pricing page

  Note: storage ratio ranges from ~3× (short docs, 19 tokens) to
  ~22× (long docs, 130 tokens). Always ask for avg tokens/doc.
```

Add an interactive version to the dashboard namespace detail page. Inputs: document
count, avg tokens/doc, dim, precision. Output: projected monthly storage cost.

**Implementation complexity:** Low (docs), Medium (dashboard widget). High customer
impact — sets expectations before the credit card bill does.

---

## Prioritization Matrix

```
                     HIGH CUSTOMER IMPACT
                            │
         P3: Server-side    │    P2: Native multi-vector docs
         MaxSim reranking   │    (correct UX, doc deletion,
         (consistent        │     top-K document results)
         scoring, no        │
         client code)       │
                            │
LOW ────────────────────────┼──────────────────────────── HIGH
EFFORT                      │                             EFFORT
                            │
         P1: Raise          │    (nothing here — low effort,
         multi-query        │     low impact options skip)
         limit to 64        │
         P4: 128-dim        │
         namespace hint     │
         P5: Cost estimator │
                            │
                     LOW CUSTOMER IMPACT
```

**Recommended sequence:**

1. **Sprint 1 (Low effort, immediate impact):** P1 (raise multi-query limit) +
   P4 (128-dim namespace hint) + P5 (cost estimator in docs). Ships within
   one sprint, removes the worst paper cuts.

2. **Sprint 2–3 (Medium effort, correctness win):** P3 (server-side MaxSim
   reranking). Eliminates the biggest source of customer bugs. Requires a fetch-by-
   doc_id query path that may have uses beyond late interaction.

3. **Long-term (High effort, architectural):** P2 (native multi-vector documents).
   Correct product abstraction. Makes turbopuffer the Vespa alternative without the
   Vespa complexity. Sequence after P3 so we have learnings on the MaxSim
   aggregation performance characteristics before committing to the storage model.

---

## Competitive Context

- **Qdrant** has [named vectors](https://qdrant.tech/documentation/concepts/collections/#collection-with-multiple-vectors)
  — multiple vectors per point with separate indexes. Close to Proposal 2 but
  indexes are separate rather than jointly scored.
- **Weaviate** has `multi2vec` — multi-modal multi-vector support. More complex
  than what's needed here.
- **Vespa** has native ColBERT support including WAND-based MaxSim approximation.
  It's powerful but requires significant operational investment to run. Turbopuffer's
  pitch is that you get competitive retrieval quality without managing infrastructure.

turbopuffer's advantage is simplicity and performance on object storage. Proposals
here are intentionally conservative on API surface: no new query languages, no
complex index configs. The goal is to make the existing approach (one row per token,
client-side rerank) require fewer steps, not to replace it with something that needs
a PhD to configure.

---

## Success Metrics

| Metric | How to measure | Target (6 months post-ship) |
|--------|---------------|----------------------------|
| Multi-vector namespace adoption | Namespaces with avg >5 vecs/doc | 20+ paying customers |
| Support ticket reduction | Tickets tagged "late interaction setup" | 50% reduction vs baseline |
| Multi-query limit utilization | P90 queries/call after limit raise | >12 (up from current ~8) |
| Query latency improvement | Avg latency for multi-vector queries | 40% reduction vs today |
| Cost estimator engagement | Dashboard widget sessions | 200+ uses/month |

The clearest leading indicator: if P3 (server-side MaxSim) ships and we see customers
migrating away from their custom reranking implementations, the feature is pulling
weight. If adoption is flat, we've learned something about whether the reranking
step is actually where friction lives.
