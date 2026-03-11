"""
Dataset loaders returning a common format for indexing and evaluation.

Both loaders produce:
  - documents  : list[{"id": str, "text": str}] — corpus to index
  - doc_store  : dict[str, str] id → text — for result hydration
  - eval_pairs : list[(query_text, target_doc_id)] — ground truth for Recall@k
  - ns_late    : turbopuffer namespace name for late-interaction index
  - ns_dense   : turbopuffer namespace name for dense index
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    ns_late: str
    ns_dense: str
    documents: list[dict]
    doc_store: dict[str, str]
    eval_pairs: list[tuple[str, str]]


def load_quora(n_pairs: int = 5000) -> DatasetConfig:
    """
    Quora Duplicate Questions.

    Documents  : unique questions (~9,800 from 5,000 pairs)
    Eval pairs : (question_text, duplicate_question_id)
    Avg tokens : ~19/doc (short questions)
    """
    from datasets import load_dataset

    print(f"Loading Quora ({n_pairs} pairs)...")
    ds = load_dataset("quora", split=f"train[:{n_pairs}]", trust_remote_code=True)

    questions: dict[int, str] = {}
    dup_pairs: list[tuple[str, str]] = []

    for ex in ds:
        ids = ex["questions"]["id"]
        texts = ex["questions"]["text"]
        for qid, qtext in zip(ids, texts):
            questions[qid] = qtext
        if ex["is_duplicate"]:
            dup_pairs.append((str(ids[0]), str(ids[1])))

    doc_store = {str(k): v for k, v in questions.items()}
    documents = [{"id": k, "text": v} for k, v in doc_store.items()]

    # eval_pairs: (query_text, target_doc_id)
    eval_pairs = [
        (doc_store[q1], q2)
        for q1, q2 in dup_pairs
        if q1 in doc_store and q2 in doc_store
    ]

    print(f"  {len(documents):,} unique questions, {len(eval_pairs):,} eval pairs")
    return DatasetConfig(
        name="quora",
        ns_late="late-interaction-quora",
        ns_dense="dense-baseline-quora",
        documents=documents,
        doc_store=doc_store,
        eval_pairs=eval_pairs,
    )


def load_squad(n_examples: int = 10_000) -> DatasetConfig:
    """
    SQuAD v1.1 (Stanford Question Answering Dataset).

    Documents  : unique Wikipedia passages (~1,400 from 10k examples)
    Eval pairs : (question, source_passage_id)
    Avg tokens : ~160/doc — roughly 8× Quora, demonstrating late interaction
                 advantage on longer, denser text.

    Retrieval task: given a question, find the passage it was written about.
    This is harder than Quora duplicate detection because the question and passage
    use different vocabulary (question asks "who", passage describes events).
    """
    from datasets import load_dataset

    print(f"Loading SQuAD ({n_examples} examples)...")
    ds = load_dataset("squad", split=f"train[:{n_examples}]")

    context_to_id: dict[str, str] = {}
    documents: list[dict] = []
    doc_store: dict[str, str] = {}
    eval_pairs: list[tuple[str, str]] = []

    for ex in ds:
        ctx = ex["context"]
        if ctx not in context_to_id:
            doc_id = f"ctx_{len(documents)}"
            context_to_id[ctx] = doc_id
            documents.append({"id": doc_id, "text": ctx})
            doc_store[doc_id] = ctx

        eval_pairs.append((ex["question"], context_to_id[ctx]))

    print(f"  {len(documents):,} unique passages, {len(eval_pairs):,} eval pairs")
    return DatasetConfig(
        name="squad",
        ns_late="late-interaction-squad",
        ns_dense="dense-baseline-squad",
        documents=documents,
        doc_store=doc_store,
        eval_pairs=eval_pairs,
    )


LOADERS = {
    "quora": load_quora,
    "squad": load_squad,
}


def load(name: str) -> DatasetConfig:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(LOADERS)}")
    return LOADERS[name]()
