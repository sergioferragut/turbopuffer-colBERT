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
| `config.py` | turbopuffer client from env vars |
| `dataset_loaders.py` | Dataset loaders for Quora and SQuAD → common `DatasetConfig` |
| `encoder.py` | `ColBERTEncoder` — token-level and dense encoding |
| `index.py` | Chunk documents and index both namespaces |
| `search.py` | `late_interaction_search` and `dense_search` |
| `evaluate.py` | Recall@10 and latency comparison |

## Run

**Step 1: Index** (~10–30 min on CPU, depending on hardware)

```bash
python index.py --dataset quora   # default
python index.py --dataset squad
```

Long documents are split into overlapping 120-word chunks before indexing.
Creates two turbopuffer namespaces per dataset:

| Namespace | Rows | Dims |
|-----------|------|------|
| `late-interaction-quora` | ~185K token rows | 128-dim ColBERT |
| `dense-baseline-quora` | ~9.9K chunk rows | 768-dim BERT |
| `late-interaction-squad` | ~293K token rows | 128-dim ColBERT |
| `dense-baseline-squad` | ~2.6K chunk rows | 768-dim BERT |

**Step 2: Evaluate**

```bash
python evaluate.py --dataset quora
python evaluate.py --dataset squad
```

Runs Recall@10 and latency benchmarks over 50 sampled query/document pairs.

| Dataset | Dense Recall@10 | Late Interaction Recall@10 |
|---------|-----------------|---------------------------|
| Quora (18.8 tokens/doc) | 0.940 | **0.960** |
| SQuAD (129.7 tokens/doc) | 0.880 | **0.980** |

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

Long documents are split into overlapping word-level chunks. Each chunk is encoded
into one 128-dim vector per token (ColBERT). All chunks share the parent `doc_id`,
so MaxSim aggregation across chunks is automatic. At query time:

1. Encode the query → Q token vectors
2. ANN-search turbopuffer for each query token (batched via `multi_query`, 16/call)
3. Accumulate per-document maximum similarity per query token
4. MaxSim score = sum of per-token maxima → re-rank candidates

See [late_interaction_guide.md](late_interaction_guide.md) for the full writeup.
