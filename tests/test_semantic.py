def test_rank_fusion_deduplicates_candidates_and_stays_within_limit():
    from hmph.semantic import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(([2, 1, 0], [1, 2, 3]), limit=3)
    assert fused[:2] == [1, 2]
    assert len(fused) == 3 and len(set(fused)) == 3


def test_public_index_reuses_vectors_and_rejects_changed_source():
    from types import SimpleNamespace

    import pytest

    from hmph.semantic import LocalSemanticRanker

    calls = []

    def passages(texts, batch_size):
        calls.append(tuple(texts))
        return iter([[1.0, 0.0], [0.0, 1.0]])

    ranker = LocalSemanticRanker.__new__(LocalSemanticRanker)
    ranker.model = SimpleNamespace(passage_embed=passages, query_embed=lambda q: iter([[1.0, 0.0]]))
    documents = [
        SimpleNamespace(
            source_id=str(i),
            session_id="public",
            timestamp="2026-11-04",
            start_offset=0,
            end_offset=1,
            content=text,
        )
        for i, text in enumerate(("a", "b"))
    ]
    ranker.prepare_public_index(documents)
    assert ranker.rank(documents, "first") == [0, 1]
    assert ranker.rank(documents, "second") == [0, 1]
    assert len(calls) == 1 and ranker.last_metrics["index_reused"]
    documents[0].content = "changed"
    with pytest.raises(ValueError, match="corpus changed"):
        ranker.rank(documents, "third")
    ranker.discard_public_index()
    assert ranker._prepared is None
