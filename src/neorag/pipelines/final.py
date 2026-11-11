"""Frozen submission profile: the already evaluated concise reranker."""

from neorag.pipelines.concise import ConciseRerankPipeline


class FinalPipeline(ConciseRerankPipeline):
    """Use the measured train-0.9048 configuration without invalidating API cache."""
