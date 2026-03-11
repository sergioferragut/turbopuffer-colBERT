# ColBERT Late Interaction on turbopuffer

Implementation of [Late Interaction Search with ColBERT on turbopuffer](late_interaction_guide.md).

## Setup

```bash
pip install -r requirements.txt
export TURBOPUFFER_API_KEY=your_key_here
export TURBOPUFFER_REGION=gcp-us-central1   # adjust to your region
```

## Files

| File | Purpose |
|------|---------|
| `encoder.py` | `ColBERTEncoder` — token-level and dense encoding |
| `index.py` | Load Quora dataset → index both namespaces |
| `search.py` | `late_interaction_search` and `dense_search` |
| `evaluate.py` | Recall@10 and latency comparison |

## Run

**Step 1: Index** (~10–30 min on CPU, depending on hardware)

```bash
python index.py
```

Creates two turbopuffer namespaces:
- `late-interaction-quora` — ~300K token rows (128-dim ColBERT embeddings)
- `dense-baseline-quora` — ~5K document rows (768-dim mean-pool BERT)

**Step 2: Evaluate**

```bash
python evaluate.py
```

Runs Recall@10 and latency benchmarks over 50 sampled duplicate pairs.

**Interactive search**

```python
from encoder import ColBERTEncoder
from search import late_interaction_search, dense_search

encoder = ColBERTEncoder()
doc_store = {"123": "What causes inflation?", ...}   # load your own

results = late_interaction_search("causes of inflation", doc_store, encoder)
for r in results:
    print(f"[{r['score']:.4f}] {r['text']}")
```

## How It Works

Each document is encoded into one 128-dim vector per token (ColBERT). These are
stored as individual turbopuffer rows keyed by `doc_id`. At query time:

1. Encode the query → Q token vectors
2. ANN-search turbopuffer for each query token (batched via `multi_query`, 16/call)
3. Accumulate per-document maximum similarity per query token
4. MaxSim score = sum of per-token maxima → re-rank candidates

See [late_interaction_guide.md](late_interaction_guide.md) for the full writeup.
