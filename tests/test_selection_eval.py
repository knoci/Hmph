import json
from pathlib import Path


def test_selection_writer_and_reader_never_receive_gold_or_category():
    from hmph.selection_eval import load_selection, raw_context, reader_messages

    cases, _ = load_selection(Path("evals/fixtures/framework-selection-dev.json"))
    assert len(cases) == 24
    case = cases[0]
    exposed = json.dumps(case.writer_input(), ensure_ascii=False)
    assert "Lumen Works" in exposed  # Normal history includes the real fact.
    for field in ("gold", "category", "rationale", "aliases", "evidence_sources", "case-001"):
        assert field not in exposed
    messages = json.dumps(reader_messages(case.query, raw_context(case)), ensure_ascii=False)
    for field in ("gold", "category", "rationale", "aliases", "evidence_sources", "case-001"):
        assert field not in messages


def test_selection_scoring_keeps_answer_evidence_and_citation_failures_separate():
    from hmph.selection_eval import Reply, grade, load_selection

    cases, _ = load_selection(Path("evals/fixtures/framework-selection-dev.json"))
    case = cases[0]
    result = grade(case, Reply(answer="Lumen Works", source_ids=("invented",)), {"s001", "s002"})
    assert result["answer_correct"] and not result["citation_valid"]
    assert not result["correct"]
    result = grade(case, Reply(answer=" lumen  works ", source_ids=("s002",)), {"s001", "s002"})
    assert result["correct"] and result["required_evidence_covered"]
    assert not grade(case, Reply(answer="I work at Lumen Works", source_ids=("s002",)), {"s002"})[
        "answer_correct"
    ]  # Freeze exact short-answer scoring; do not silently relax it after seeing predictions.


def test_blind_source_errata_preserves_answers_and_model_inputs():
    from hmph.selection_diagnostics import corrected_cases
    from hmph.selection_eval import Reply, grade, load_selection

    cases, checksum = load_selection(Path("evals/fixtures/framework-selection-dev.json"))
    errata = json.loads(
        Path("evals/fixtures/framework-selection-source-errata-v1.json").read_text(encoding="utf-8")
    )
    revised = corrected_cases(cases, checksum, errata)
    for old, new in zip(cases, revised, strict=True):
        assert old.writer_input() == new.writer_input() and old.query == new.query
        assert old.gold.answer == new.gold.answer and old.gold.aliases == new.gold.aliases
    old, new = cases[3], revised[3]
    reply = Reply(answer=old.gold.answer, source_ids=("s001",))
    assert old.gold.evidence_sources == ("s001", "s002")
    assert not grade(old, reply, {"s001"})["correct"]
    assert grade(new, reply, {"s001"})["correct"]
