"""
Evaluate late interaction vs dense retrieval.

Metrics:
  - Recall@10 : does the known relevant document appear in top-10 results?
  - Latency   : avg and P95 query time over sampled queries

Usage:
    python evaluate.py --dataset quora   # default
    python evaluate.py --dataset squad
"""

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

from config import client
from dataset_loaders import DatasetConfig, load
from encoder import ColBERTEncoder, BERT_DIM, DIM
from search import late_interaction_search, dense_search

EVAL_SAMPLE = 50
RANDOM_SEED = 42
TOP_K = 10


# ── result tracking ────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    name: str
    hits: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return self.hits / max(len(self.latencies_ms), 1)

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]


# ── warm-up ────────────────────────────────────────────────────────────────


def warm_caches(cfg: DatasetConfig) -> None:
    """Issue dummy queries to prime turbopuffer's SSD cache."""
    print("Warming turbopuffer caches...", end=" ", flush=True)
    client.namespace(cfg.ns_late).query(rank_by=("vector", "ANN", [0.0] * DIM), top_k=1)
    client.namespace(cfg.ns_dense).query(rank_by=("vector", "ANN", [0.0] * BERT_DIM), top_k=1)
    print("done.")


# ── evaluation loop ────────────────────────────────────────────────────────


def run_eval(
    encoder: ColBERTEncoder,
    cfg: DatasetConfig,
    n_samples: int = EVAL_SAMPLE,
) -> tuple[EvalResult, EvalResult]:
    random.seed(RANDOM_SEED)

    sample = random.sample(cfg.eval_pairs, min(n_samples, len(cfg.eval_pairs)))

    late_res = EvalResult("Late Interaction (ColBERT MaxSim)")
    dense_res = EvalResult("Dense (BERT mean-pool, 768-dim)")

    print(f"\nEvaluating on {len(sample)} pairs (Recall@{TOP_K})...")
    print(f"{'#':>4}  {'Method':<12}  {'Hit':>4}  {'ms':>6}  Query (truncated)")
    print("-" * 72)

    for i, (query_text, target_doc_id) in enumerate(sample):
        # Late interaction
        t0 = time.perf_counter()
        li_results = late_interaction_search(query_text, cfg.doc_store, encoder,
                                             top_k=TOP_K, ns=client.namespace(cfg.ns_late))
        li_ms = (time.perf_counter() - t0) * 1000
        li_hit = any(r["doc_id"] == target_doc_id for r in li_results)
        late_res.hits += int(li_hit)
        late_res.latencies_ms.append(li_ms)

        # Dense
        t0 = time.perf_counter()
        d_results = dense_search(query_text, cfg.doc_store, encoder,
                                 top_k=TOP_K, ns=client.namespace(cfg.ns_dense))
        d_ms = (time.perf_counter() - t0) * 1000
        d_hit = any(r["doc_id"] == target_doc_id for r in d_results)
        dense_res.hits += int(d_hit)
        dense_res.latencies_ms.append(d_ms)

        label = query_text[:50].replace("\n", " ")
        print(f"{i+1:>4}  {'late':12}  {'✓' if li_hit else '✗':>4}  {li_ms:>6.1f}  {label}")
        print(f"{'':>4}  {'dense':12}  {'✓' if d_hit else '✗':>4}  {d_ms:>6.1f}")

    return late_res, dense_res


# ── qualitative examples ───────────────────────────────────────────────────


def show_qualitative(
    encoder: ColBERTEncoder,
    cfg: DatasetConfig,
    n: int = 3,
) -> None:
    """Show examples where the two methods disagree."""
    random.seed(RANDOM_SEED + 1)
    sample = random.sample(cfg.eval_pairs, min(300, len(cfg.eval_pairs)))

    shown = 0
    print("\n=== Qualitative Examples ===")

    for query_text, target_doc_id in sample:
        if shown >= n:
            break

        li_results = late_interaction_search(query_text, cfg.doc_store, encoder,
                                             top_k=TOP_K, ns=client.namespace(cfg.ns_late))
        d_results = dense_search(query_text, cfg.doc_store, encoder,
                                 top_k=TOP_K, ns=client.namespace(cfg.ns_dense))

        li_hit = any(r["doc_id"] == target_doc_id for r in li_results)
        d_hit = any(r["doc_id"] == target_doc_id for r in d_results)

        if li_hit == d_hit:
            continue

        winner = "late interaction" if li_hit else "dense"
        target_text = cfg.doc_store.get(target_doc_id, "")
        print(f"\nExample {shown + 1} (winner: {winner})")
        print(f"  Query      : {query_text}")
        print(f"  Target doc : {target_text[:120]}...")
        print(f"  Late top-1 : {li_results[0]['text'][:120] if li_results else 'n/a'}...")
        print(f"  Dense top-1: {d_results[0]['text'][:120] if d_results else 'n/a'}...")
        shown += 1

    if shown == 0:
        print("  (no disagreements found in sample — methods agree on this subset)")


# ── report ────────────────────────────────────────────────────────────────


def print_report(late: EvalResult, dense: EvalResult, cfg: DatasetConfig) -> None:
    n = len(late.latencies_ms)
    print("\n" + "=" * 60)
    print(f"RESULTS — {cfg.name.upper()}")
    print("=" * 60)
    print(f"\nRecall@{TOP_K} ({n} queries):")
    print(f"  {late.name:<45} {late.recall_at_k:.3f}  ({late.hits}/{n})")
    print(f"  {dense.name:<45} {dense.recall_at_k:.3f}  ({dense.hits}/{n})")
    print(f"\nLatency (ms):")
    print(f"  {'Method':<45}  {'Avg':>8}  {'P95':>8}")
    print(f"  {late.name:<45}  {late.avg_latency_ms:>8.1f}  {late.p95_latency_ms:>8.1f}")
    print(f"  {dense.name:<45}  {dense.avg_latency_ms:>8.1f}  {dense.p95_latency_ms:>8.1f}")
    print(f"\nRecall improvement : {late.recall_at_k - dense.recall_at_k:+.3f}")
    print(f"Latency overhead   : {late.avg_latency_ms / max(dense.avg_latency_ms, 0.01):.1f}×")


# ── entrypoint ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="quora", choices=["quora", "squad"],
                        help="Dataset to evaluate (default: quora)")
    args = parser.parse_args()

    cfg = load(args.dataset)
    encoder = ColBERTEncoder()

    warm_caches(cfg)

    late_result, dense_result = run_eval(encoder, cfg)
    show_qualitative(encoder, cfg)
    print_report(late_result, dense_result, cfg)
