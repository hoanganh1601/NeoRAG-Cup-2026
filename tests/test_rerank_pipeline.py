import json

import numpy as np
import pytest

from neorag.data import Document
from neorag.pipelines.concise import ConciseRerankPipeline
from neorag.pipelines.focused import FocusedRerankPipeline
from neorag.pipelines.hybrid import HybridPipeline
from neorag.pipelines.rerank import RerankPipeline


class UniformEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)


class FakeRerankLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, model):
        self.calls.append(messages)
        if messages[0]["role"] == "system":
            return self.reply
        return "Câu trả lời thử nghiệm."


def make_pipeline(reply, *, count=25, **kwargs):
    llm = FakeRerankLLM(reply)
    pipeline = RerankPipeline(llm=llm, embedder=UniformEmbedder(), **kwargs)
    pipeline.index(
        [
            Document(100 + i, "CLB Lập Trình PTIT", "Giới thiệu", f"Nội dung đoạn {i}.")
            for i in range(count)
        ]
    )
    return pipeline, llm


def test_hybrid_candidate_pool_can_expand_without_changing_default_ranking():
    pipeline, llm = make_pipeline("{}")
    normal = HybridPipeline.rank(pipeline, "ProPTIT")
    expanded = HybridPipeline.rank(pipeline, "ProPTIT", candidate_k=20)
    assert len(normal) == 10
    assert len(expanded) == 20
    assert expanded[:10].tolist() == normal.tolist()
    assert pipeline.n_retrieved == 10
    assert not llm.calls


def test_reranking_filters_invented_ids_duplicates_and_noninteger_values():
    pipeline, llm = make_pipeline('{"ranked_ids": [119, 103, 119, 999, true, 104.0, "105", 100]}')
    result = pipeline.run("Thông tin ProPTIT")
    assert result.retrieved_ids[:3] == [119, 103, 100]
    assert len(result.retrieved_ids) == len(set(result.retrieved_ids)) == 10
    assert set(result.retrieved_ids) <= set(range(100, 120))
    assert result.answer == "Câu trả lời thử nghiệm."
    assert len(llm.calls) == 2  # one rerank and one answer, with no real gateway
    rerank_input = json.loads(llm.calls[0][1]["content"])
    assert len(rerank_input["candidates"]) == 20
    assert "ranked_ids" not in rerank_input
    answer_prompt = llm.calls[1][0]["content"]
    assert answer_prompt.index("[Đoạn 119") < answer_prompt.index("[Đoạn 103")
    assert answer_prompt.count("[Đoạn ") == 7


@pytest.mark.parametrize("reply", ["not JSON", '{"other": []}', '{"ranked_ids": null}'])
def test_invalid_response_falls_back_to_hybrid_without_an_extra_request(reply):
    pipeline, llm = make_pipeline(reply)
    expected = HybridPipeline.rank(pipeline, "ProPTIT").tolist()
    with pytest.warns(UserWarning, match="invalid JSON"):
        actual = pipeline.rank("ProPTIT")
    assert actual.tolist() == expected
    assert len(llm.calls) == 1


@pytest.mark.parametrize("reply", ['{"ranked_ids": []}', '{"ranked_ids": [999, true]}'])
def test_no_valid_ids_falls_back_to_original_candidates(reply):
    pipeline, _ = make_pipeline(reply)
    expected = HybridPipeline.rank(pipeline, "ProPTIT").tolist()
    with pytest.warns(UserWarning, match="no valid candidate IDs"):
        actual = pipeline.rank("ProPTIT")
    assert actual.tolist() == expected


def test_fenced_json_is_accepted_and_short_response_is_completed_in_hybrid_order():
    pipeline, _ = make_pipeline('```json\n{"ranked_ids": [103]}\n```', count=5)
    result = pipeline.run("ProPTIT")
    assert result.retrieved_ids == [103, 100, 101, 102, 104]


def test_single_document_skips_unnecessary_reranking():
    pipeline, llm = make_pipeline("unused", count=1)
    result = pipeline.run("ProPTIT")
    assert result.retrieved_ids == [100]
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "user"


@pytest.mark.parametrize(
    "kwargs", [{"candidate_k": 0}, {"candidate_k": 5}, {"n_retrieved": 11}, {"candidate_k": True}]
)
def test_invalid_candidate_or_submission_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_pipeline("{}", **kwargs)


@pytest.mark.parametrize("pipeline_cls", [FocusedRerankPipeline, ConciseRerankPipeline])
def test_answer_prompt_experiment_preserves_rerank_request_and_source_order(pipeline_cls):
    original, original_llm = make_pipeline('{"ranked_ids": [103, 101]}')
    trial_llm = FakeRerankLLM('{"ranked_ids": [103, 101]}')
    trial = pipeline_cls(llm=trial_llm, embedder=UniformEmbedder())
    trial.index(original.docs)
    previous = original.run("ProPTIT")
    current = trial.run("ProPTIT")
    # Identical rerank messages/model retain the existing LLM cache key.
    assert original_llm.calls[0] == trial_llm.calls[0]
    assert previous.retrieved_ids == current.retrieved_ids
    assert len(original_llm.calls) == len(trial_llm.calls) == 2
    assert original_llm.calls[1] != trial_llm.calls[1]
    assert (
        original_llm.calls[1][0]["content"].split("TÀI LIỆU:\n", 1)[1]
        == (trial_llm.calls[1][0]["content"].split("TÀI LIỆU:\n", 1)[1])
    )
