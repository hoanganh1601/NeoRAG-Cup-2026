"""Compare retrieval settings on train, using cached embeddings only (no API calls)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from dotenv import load_dotenv

from neorag.data import DATA_DIR, load_corpus, load_queries, write_jsonl
from neorag.llm import _key
from neorag.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from neorag.pipelines.hybrid import HybridPipeline


class CachedEmbedder:
    """A cache miss raises an error rather than spending API budget."""

    def __init__(self, cache: Path):
        self.connection = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
        self.endpoint = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/"
        self.model = os.getenv("EMBEDDING_MODEL", "bge-m3")

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            key = _key("embed", self.endpoint, self.model, batch)
            hit = self.connection.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if hit is None:
                raise ValueError(f"Embedding not cached for batch of {len(batch)} texts")
            rows.extend(json.loads(hit[0]))
        vectors = np.asarray(rows, dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def metrics(rankings: list[list[int]], queries) -> dict:
    included = [(rank, q) for rank, q in zip(rankings, queries, strict=True) if q.relevant_ids]
    values = {
        name: mean(fn(rank, set(q.relevant_ids), 5) for rank, q in included)
        for name, fn in (
            ("recall@5", recall_at_k),
            ("mrr@5", reciprocal_rank_at_k),
            ("ndcg@5", ndcg_at_k),
        )
    }
    values["retrieval_score"] = mean(values.values())
    return values


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/retrieval_experiments.jsonl"))
    args = parser.parse_args()
    corpus = load_corpus()
    queries = load_queries(DATA_DIR / "train/questions.jsonl")
    embedder = CachedEmbedder(Path(".cache/pipeline.sqlite"))
    query_vectors = np.asarray([embedder.embed([q.query])[0] for q in queries])
    baseline = HybridPipeline(llm=object(), embedder=embedder)
    baseline.index(corpus)
    from neorag.text import tokenize

    old_rankings = []
    for query, vector in zip(queries, query_vectors, strict=True):
        dense = baseline.doc_vectors @ vector
        dense = (dense - dense.min()) / max(float(np.ptp(dense)), 1e-8)
        tokens = set(tokenize(query.query))
        denominator = max(sum(baseline.idf.get(t, 0) for t in tokens), 1e-8)
        lexical = np.asarray(
            [
                sum(baseline.idf.get(t, 0) for t in tokens & doc) / denominator
                for doc in baseline.doc_token_sets
            ]
        )
        order = np.argsort(-(0.85 * dense + 0.15 * lexical))[:10]
        old_rankings.append([corpus[i].id for i in order])
    print("Current hybrid:", json.dumps(metrics(old_rankings, queries)))
    experiments = []
    for bigrams in (False, True):
        pipeline = HybridPipeline(llm=object(), embedder=embedder, bigrams=bigrams)
        pipeline.index(corpus)
        dense_scores = baseline.doc_vectors @ query_vectors.T
        dense_scores = (dense_scores - dense_scores.min(axis=0)) / np.maximum(
            np.ptp(dense_scores, axis=0), 1e-8
        )
        lexical_scores = np.asarray([pipeline.lexical_scores(q.query) for q in queries]).T
        from neorag.pipelines.hybrid import mentioned_clubs

        bonus = np.asarray(
            [
                [bool(mentioned_clubs(q.query) & club) for club in pipeline.doc_clubs]
                for q in queries
            ]
        ).T
        for weight in (0.15, 0.3, 0.45, 0.6):
            for club_bonus in (0.0, 0.1, 0.2):
                scores = (1 - weight) * dense_scores + weight * lexical_scores + club_bonus * bonus
                rankings = [
                    [corpus[i].id for i in np.argsort(-scores[:, j], kind="stable")[:10]]
                    for j in range(len(queries))
                ]
                result = {
                    "bigrams": bigrams,
                    "lexical_weight": weight,
                    "club_bonus": club_bonus,
                    **metrics(rankings, queries),
                }
                # These are development diagnostics, not held-out test results.
                answerable = [
                    (r, q) for r, q in zip(rankings, queries, strict=True) if q.answerable
                ]
                result["answerable_folds"] = [
                    metrics(
                        [r for r, _ in answerable[offset::3]], [q for _, q in answerable[offset::3]]
                    )["retrieval_score"]
                    for offset in range(3)
                ]
                experiments.append(result)
    experiments.sort(key=lambda row: row["retrieval_score"], reverse=True)
    write_jsonl(args.out, experiments)
    for row in experiments[:8]:
        print(json.dumps(row))
    candidate = HybridPipeline(llm=object(), embedder=embedder)
    candidate.index(corpus)
    rankings = [
        [corpus[i].id for i in candidate.rank(query.query, vector)]
        for query, vector in zip(queries, query_vectors, strict=True)
    ]
    write_jsonl(
        args.out.with_name("retrieval_candidate_per_query.jsonl"),
        [
            {
                "id": query.id,
                "old_retrieved_ids": old,
                "new_retrieved_ids": new,
                "relevant_ids": query.relevant_ids,
            }
            for query, old, new in zip(queries, old_rankings, rankings, strict=True)
        ],
    )
    print("Selected defaults:", json.dumps(metrics(rankings, queries)))


if __name__ == "__main__":
    main()
