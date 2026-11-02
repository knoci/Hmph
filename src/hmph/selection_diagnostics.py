"""Apply independently audited source-label errata without changing predictions.

This creates a separate diagnostic. It never rewrites the frozen primary scores,
calls a model, changes an answer/alias, or changes any model-facing input.
"""

import copy
import hashlib
import json
from dataclasses import replace

from hmph.selection_eval import Reply, grade, load_selection, screen_profiles


def corrected_cases(cases, checksum, errata):
    if checksum != errata["dataset_sha256"]:
        raise ValueError("Errata dataset mismatch")
    corrected = []
    observed = set()
    for case in cases:
        correction = errata["corrections"].get(case.case_id)
        if correction is not None:
            if list(case.gold.evidence_sources) != correction["before"] or not set(
                correction["after"]
            ).issubset({source.source_id for source in case.history}):
                raise ValueError("Invalid source erratum")
            case = replace(
                case,
                gold=case.gold.model_copy(
                    update={
                        "evidence_sources": tuple(correction["after"]),
                    }
                ),
            )
            observed.add(case.case_id)
        corrected.append(case)
    if observed != set(errata["corrections"]):
        raise ValueError("Unknown source erratum")
    return corrected


def diagnose(directory, suite, errata_path):
    primary_path = directory / "report.json"
    raw = primary_path.read_bytes()
    primary = json.loads(raw)
    cases, checksum = load_selection(suite)
    errata = json.loads(errata_path.read_text(encoding="utf-8"))
    cases = corrected_cases(cases, checksum, errata)
    by_id = {case.case_id: case for case in cases}
    profiles = copy.deepcopy(primary["profiles"])
    for value in profiles.values():
        for row in value.get("cases", []):
            if row["status"] != "completed":
                continue
            reply = Reply.model_validate_json(json.dumps(row["reply"]))
            supplied = {sid for item in row["evidence"] for sid in item["source_ids"]}
            scores = grade(by_id[row["case_id"]], reply, supplied)
            if scores["answer_correct"] != row["scores"]["answer_correct"]:
                raise ValueError("Source errata must not change answer accuracy")
            row["scores"] = scores
    output = directory / "diagnostics-source-errata.json"
    if output.exists():
        raise ValueError("Do not overwrite the diagnostic")
    result = {
        "kind": "posthoc_source_label_errata",
        "primary_report_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": checksum,
        "errata": errata,
        "primary_screening": primary["screening"],
        "corrected_screening": screen_profiles(profiles, cases),
        "all_predictions_reused": True,
        "answer_accuracy_unchanged": True,
        "additional_model_calls": 0,
        "corrected_rows": {
            name: [
                {"case_id": row["case_id"], "scores": row.get("scores")}
                for row in value.get("cases", [])
            ]
            for name, value in profiles.items()
        },
        "limitation": "Posthoc audited diagnostic, not a replacement for preregistered scores.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if primary_path.read_bytes() != raw:
        raise RuntimeError("Primary report changed during diagnostics")
    return result
