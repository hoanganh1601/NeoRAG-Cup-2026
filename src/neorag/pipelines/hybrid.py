"""Hybrid search with accent-insensitive BM25 and soft club routing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

from neorag.data import Document
from neorag.pipelines.base import RAGOutput
from neorag.pipelines.baseline import PROMPT, BaselinePipeline

_STOPWORDS = {
    "la",
    "va",
    "cua",
    "cac",
    "nhung",
    "mot",
    "duoc",
    "cho",
    "khi",
    "thi",
    "co",
    "khong",
    "voi",
    "em",
    "e",
    "a",
    "ak",
    "ko",
    "k",
    "dc",
    "gi",
    "nao",
    "vay",
    "hoi",
    "minh",
    "toi",
    "ah",
    "nhe",
    "the",
    "nay",
    "do",
    "o",
    "hay",
    "se",
    "rat",
    "muon",
}
_CLUB_PATTERNS = {
    "programming": r"\b(?:proptit|clb lap trinh(?: ptit)?|clb lt(?: ptit)?|programming ptit)\b",
    "multimedia": r"\b(?:mpc|multimedia(?: ptit)?|clb mm|clb da phuong tien)\b",
    "game": r"\b(?:gcc|game creators club|(?:clb |cau lac bo )?(?:nha sang tao game|lam game)|clb game)\b",
}
_ALIASES = (
    (r"\b(?:proptit|clb lt(?: ptit)?)\b", "clb lap trinh ptit"),
    (r"\b(?:gcc|game creators club|clb game)\b", "clb nha sang tao game ptit"),
    (r"\b(?:mpc|clb mm|clb da phuong tien)\b", "clb multimedia ptit"),
)


def fold_text(text: str) -> str:
    """Casefold and remove Vietnamese accents for keyword matching only."""
    text = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def mentioned_clubs(text: str) -> set[str]:
    folded = fold_text(text)
    return {club for club, pattern in _CLUB_PATTERNS.items() if re.search(pattern, folded)}


def lexical_tokens(text: str, *, bigrams: bool = True, aliases: bool = True) -> list[str]:
    folded = fold_text(text)
    if aliases:
        for pattern, replacement in _ALIASES:
            folded = re.sub(pattern, replacement, folded)
    raw = folded.split()
    tokens = [word for word in raw if word not in _STOPWORDS]
    if bigrams:
        tokens += [f"{left}_{right}" for left, right in zip(raw, raw[1:], strict=False)]
    return tokens


def scaled(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / max(float(np.ptp(scores)), 1e-8)


class HybridPipeline(BaselinePipeline):
    """Keep the user's generation settings; improve retrieval independently."""

    answer_prompt = PROMPT

    def __init__(
        self,
        *,
        lexical_weight: float = 0.6,
        club_bonus: float = 0.2,
        bigrams: bool = True,
        aliases: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("top_k", 7)
        kwargs.setdefault("n_retrieved", 10)
        super().__init__(**kwargs)
        if not 0 <= lexical_weight <= 1 or club_bonus < 0:
            raise ValueError("lexical_weight must be in [0, 1]; club_bonus must be nonnegative")
        self.lexical_weight = lexical_weight
        self.club_bonus = club_bonus
        self.bigrams = bigrams
        self.aliases = aliases

    def index(self, corpus: list[Document]) -> None:
        super().index(corpus)
        self.term_counts = [
            Counter(lexical_tokens(doc.text, bigrams=self.bigrams, aliases=self.aliases))
            for doc in corpus
        ]
        self.doc_lengths = np.asarray([sum(terms.values()) for terms in self.term_counts])
        self.average_length = max(float(self.doc_lengths.mean()), 1.0)
        frequency = Counter(term for terms in self.term_counts for term in terms)
        self.bm25_idf = {
            term: np.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }
        self.doc_clubs = [mentioned_clubs(doc.club) for doc in corpus]

    def lexical_scores(self, query: str) -> np.ndarray:
        tokens = set(lexical_tokens(query, bigrams=self.bigrams, aliases=self.aliases))
        scores = np.zeros(len(self.docs), dtype=np.float64)
        length_norm = 1.2 * (0.25 + 0.75 * self.doc_lengths / self.average_length)
        for term in tokens:
            frequency = np.asarray([terms.get(term, 0) for terms in self.term_counts])
            scores += self.bm25_idf.get(term, 0.0) * frequency * 2.2 / (frequency + length_norm)
        return scaled(scores)

    def rank(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        *,
        candidate_k: int | None = None,
    ) -> np.ndarray:
        if query_vector is None:
            query_vector = self.embedder.embed([query])[0]
        dense = scaled(self.doc_vectors @ query_vector)
        lexical = self.lexical_scores(query)
        scores = (1 - self.lexical_weight) * dense + self.lexical_weight * lexical
        clubs = mentioned_clubs(query)
        if clubs:
            scores += self.club_bonus * np.asarray(
                [bool(clubs & doc_clubs) for doc_clubs in self.doc_clubs]
            )
        limit = self.n_retrieved if candidate_k is None else candidate_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("candidate_k must be a positive integer")
        return np.argsort(-scores, kind="stable")[:limit]

    def run(self, query: str) -> RAGOutput:
        ranking = self.rank(query)
        docs = [self.docs[i] for i in ranking]
        # Labels help the model separate different clubs and periods without
        # changing the original corpus or embedding input.
        context = "\n\n".join(
            f"[Đoạn {doc.id} | {doc.club} | {doc.topic}]\n{doc.text}" for doc in docs[: self.top_k]
        )
        answer = self.llm.chat(
            [{"role": "user", "content": self.answer_prompt.format(context=context, query=query)}],
            model=self.llm_model,
        )
        return RAGOutput(answer=answer, retrieved_ids=[doc.id for doc in docs])
