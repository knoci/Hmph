def test_rank_fusion_deduplicates_candidates_and_stays_within_limit():
    from hmph.semantic import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(([2, 1, 0], [1, 2, 3]), limit=3)
    assert fused[:2] == [1, 2]
    assert len(fused) == 3 and len(set(fused)) == 3
