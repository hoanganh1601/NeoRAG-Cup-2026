"""Baseline chính thức: naive RAG đơn giản nhất có thể.

1. Embed nguyên từng đoạn văn của corpus (không chunking) bằng bge-m3 qua gateway của BTC.
2. Với mỗi câu hỏi, lấy các đoạn có cosine similarity cao nhất.
3. Đưa top-k đoạn vào một prompt đơn giản và gọi LLM một lần.

Baseline cố ý không tối ưu gì để bạn có nhiều chỗ cải thiện, ví dụ: chunking, hybrid search
(BM25 + dense), reranker, query rewriting, lọc theo CLB, prompt tốt hơn, xử lý câu hỏi
không có đáp án, ...
"""

from __future__ import annotations
from collections import Counter
from neorag.text import tokenize

import os

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline

PROMPT = """Bạn là trợ lý hỏi đáp về các câu lạc bộ thuộc cộng đồng S2B tại PTIT.

Quy ước tên gọi, đã được xác nhận trong bộ tài liệu:
- ProPTIT, PROPTIT và CLB Lập Trình PTIT là cùng một câu lạc bộ.
- MPC, Multimedia PTIT Club và CLB Multimedia PTIT là cùng một câu lạc bộ.
- GCC, Game Creators Club và CLB Nhà Sáng Tạo Game PTIT là cùng một câu lạc bộ.
- Khi câu hỏi dùng tên viết tắt, hãy sử dụng thông tin của tên đầy đủ tương ứng trong tài liệu để trả lời.

Nhiệm vụ:
- Trả lời trực tiếp và chính xác câu hỏi của người dùng.
- Chỉ sử dụng thông tin có trong TÀI LIỆU.
- Không suy đoán, không bịa thêm thông tin.
- Nếu tài liệu không đủ để trả lời, hãy nói rõ:
  "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
- Giữ nguyên tên riêng, ngày tháng, con số và các mốc thời gian trong tài liệu.
- Nếu câu hỏi có nhiều ý, trả lời đủ từng ý.
- Trả lời ngắn gọn, tập trung đúng trọng tâm câu hỏi.
- Không nhắc đến "tài liệu", "context", prompt hoặc cách bạn suy luận.
- Không mở đầu bằng các câu chung chung như "Dựa vào tài liệu..." hoặc "Theo thông tin được cung cấp...".

TÀI LIỆU:
{context}

CÂU HỎI:
{query}

CÂU TRẢ LỜI:"""


class GatewayEmbedder:
    """Embedding qua gateway (cùng key với LLM). Muốn chạy model local thì dùng
    neorag.embeddings.LocalEmbedder (cài thêm: python scripts/install.py --local)."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận (len(texts), dim) đã chuẩn hóa L2."""
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class BaselinePipeline(RAGPipeline):
    def __init__(
        self,
        top_k: int = 7,
        n_retrieved: int = 10,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        self.top_k = top_k  # số doc đưa vào prompt
        self.n_retrieved = n_retrieved  # số doc trả về để chấm retrieval
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = embedder or GatewayEmbedder(
            self.llm, embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        )

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])

        # Chuẩn bị dữ liệu cho tìm kiếm từ khóa.
        self.doc_token_sets = [set(tokenize(d.text)) for d in corpus]

        document_frequency = Counter(
            token
            for tokens in self.doc_token_sets
            for token in tokens
        )

        number_of_docs = len(corpus)
        self.idf = {
            token: np.log((number_of_docs + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    def run(self, query: str) -> RAGOutput:
        q = self.embedder.embed([query])[0]

        # Điểm tìm kiếm ngữ nghĩa bằng embedding.
        dense_scores = self.doc_vectors @ q
        dense_scores = (dense_scores - dense_scores.min()) / (
            np.ptp(dense_scores) + 1e-8
        )
        # Điểm khớp từ khóa, ưu tiên những từ hiếm.
        query_tokens = set(tokenize(query))
        query_weight = sum(self.idf.get(token, 0.0) for token in query_tokens)
        
        lexical_scores = np.array(
            [
                sum(
                    self.idf.get(token, 0.0)
                    for token in query_tokens & doc_tokens
                ) / max(query_weight, 1e-8)
                for doc_tokens in self.doc_token_sets
            ],
            dtype=np.float32,
        )
        
        # Kết hợp semantic search và keyword search.
        hybrid_scores = 0.85 * dense_scores + 0.15 * lexical_scores
        
        ranking = np.argsort(-hybrid_scores)[: self.n_retrieved] 
               
        docs = [self.docs[i] for i in ranking]
        context = "\n\n".join(d.text for d in docs[: self.top_k])
        answer = self.llm.chat(
            [{"role": "user", "content": PROMPT.format(context=context, query=query)}],
            model=self.llm_model,
        )
        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])
