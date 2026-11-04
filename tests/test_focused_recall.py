"""The retrieval bridge is a literal navigation hint, never model-authored evidence."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hmph.working_memory import focused_query


def test_focus_requires_a_literal_span_from_the_selected_source():
    sources = [SimpleNamespace(source_id="s1", content="My sister Lucie works in Lyon.")]
    payload = {"bridge": {"source_id": "s1", "quote": "Lucie", "slot": "entity"}}
    assert focused_query("Where is her bank?", payload, sources) == (
        "Where is her bank? Lucie",
        "s1",
    )
    payload["bridge"]["quote"] = "Paris"
    with pytest.raises(ValueError, match="Unverified"):
        focused_query("Where is her bank?", payload, sources)
    payload["bridge"].update(source_id="other", quote="Lucie")
    with pytest.raises(ValueError, match="Unverified"):
        focused_query("Where is her bank?", payload, sources)


def test_focus_cannot_return_an_answer_or_arbitrary_query():
    for payload in ({"bridge": None, "answer": "Paris"}, {"query": "DROP TABLE"}):
        with pytest.raises(ValidationError):
            focused_query("Where?", payload, [])
    assert focused_query("Where?", {"bridge": None}, []) == ("Where?", None)


def test_focus_uses_only_selected_excerpt_and_enforces_total_query_limit():
    source = SimpleNamespace(source_id="s1", content="Lucie works in Lyon.")
    payload = {"bridge": {"source_id": "s1", "quote": "Lucie", "slot": "relation"}}
    with pytest.raises(ValueError, match="input limit"):
        focused_query("x" * 4000, payload, [source])
