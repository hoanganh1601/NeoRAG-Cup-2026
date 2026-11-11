import numpy as np

from neorag.data import Document
from neorag.pipelines.hybrid import HybridPipeline, fold_text, lexical_tokens, mentioned_clubs


class UniformEmbedder:
    """Remove dense differences so the tests isolate retrieval behavior."""

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)


class RecordingLLM:
    def chat(self, messages, model):
        self.messages = messages
        return "Câu trả lời thử nghiệm."


def test_vietnamese_keywords_and_aliases_match_without_accents():
    assert fold_text("ĐỐI NGOẠI") == "doi ngoai"
    assert lexical_tokens("CLB Lập Trình PTIT") == lexical_tokens("ProPTIT")
    assert mentioned_clubs("clb lt ptit với GCC") == {"programming", "game"}


def test_hybrid_ranks_matching_club_and_returns_the_same_context_sources():
    docs = [
        Document(31, "CLB Nhà Sáng Tạo Game PTIT (GCC)", "Đào tạo", "Đào tạo lập trình game."),
        Document(72, "CLB Lập Trình PTIT", "Nội quy", "Đổi lịch học phải báo trước một ngày."),
        Document(73, "CLB Lập Trình PTIT", "Nội quy", "Thành viên trả lời thông báo trên nhóm."),
    ]
    llm = RecordingLLM()
    pipeline = HybridPipeline(llm=llm, embedder=UniformEmbedder(), top_k=2, n_retrieved=2)
    pipeline.index(docs)
    result = pipeline.run("clb lt ptit doi lich hoc phai bao truoc bao lau")
    assert result.retrieved_ids[0] == 72
    assert len(result.retrieved_ids) == len(set(result.retrieved_ids)) == 2
    assert result.answer == "Câu trả lời thử nghiệm."
    assert "[Đoạn 72 | CLB Lập Trình PTIT | Nội quy]" in llm.messages[0]["content"]


def test_multiclub_question_does_not_exclude_either_club():
    docs = [
        Document(10, "CLB Lập Trình PTIT", "Liên hệ", "Email liên hệ lập trình."),
        Document(20, "CLB Nhà Sáng Tạo Game PTIT (GCC)", "Liên hệ", "Email liên hệ game."),
        Document(30, "CLB Multimedia PTIT", "Liên hệ", "Email liên hệ multimedia."),
    ]
    pipeline = HybridPipeline(llm=RecordingLLM(), embedder=UniformEmbedder(), n_retrieved=3)
    pipeline.index(docs)
    ranking = pipeline.rank("Email ProPTIT và GCC")
    assert {docs[i].id for i in ranking[:2]} == {10, 20}
