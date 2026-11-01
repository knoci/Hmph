"""Run pinned native research pipelines in the isolated Python 3.11 environment.

This is a synthetic compatibility check, not a paper reproduction score.
Run with artifacts/reference-env/Scripts/python.exe, from the repository root.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parents[1]
# The native Python 3.11 environment intentionally does not install Hmph's runtime.
sys.path.insert(0, str(ROOT / "src"))
from hmph.reference_transport import ReferenceTransport  # noqa: E402

REVISIONS = {
    "simplemem": ("SimpleMem", "db80b6a7c591e0ea730a058e9f5fc4eb06572299"),
    "lightmem": ("LightMem", "33c4d200652faae12d0577cd0cee897bc08a9e85"),
}
HISTORY = [
    ("2026-10-01T10:00:00", "My name is Mira. I work at Cedar Atelier."),
    (
        "2026-10-15T10:00:00",
        "I changed jobs. I now work at Juniper Works. I prefer unsweetened oolong tea.",
    ),
]
QUERY = "As of 2026-11-01, what is my name, current employer, and preferred drink?"
PLACEHOLDER = "harmless-native-placeholder-not-a-credential"


def run(reference, output):
    values = dotenv_values(ROOT / ".env.local")
    secret = os.getenv("HMPH_DEEPSEEK_API_KEY") or values.get("HMPH_DEEPSEEK_API_KEY")
    model = os.getenv("HMPH_DEEPSEEK_MODEL") or values.get("HMPH_DEEPSEEK_MODEL", "deepseek-flash")
    if not secret or model not in {"deepseek-flash", "deepseek-v4-pro"}:
        raise RuntimeError("Local DeepSeek configuration required")
    folder, revision = REVISIONS[reference]
    repo = ROOT / "artifacts/references" / folder
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise RuntimeError("Reference revision mismatch")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet"]).returncode:
        raise RuntimeError("Native reference source has local changes")
    run_dir = ROOT / "artifacts/references/runs" / (reference + "-" + uuid4().hex)
    run_dir.mkdir(parents=True)
    embedding_path = ROOT / "artifacts/models/all-minilm-native"
    if not (embedding_path / "config.json").exists():
        raise RuntimeError("Prepare the local native embedding model first")
    transport = ReferenceTransport(SecretStr(secret), model=model, max_calls=12)
    captured = io.StringIO()
    report = {
        "reference": reference,
        "repository_revision": revision,
        "mode": "native_pipeline_synthetic_smoke",
        "benchmark": False,
        "paper_score": None,
        "native_source_modified": False,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transport_sha256": hashlib.sha256(
            (ROOT / "src/hmph/reference_transport.py").read_bytes()
        ).hexdigest(),
        "status": "failed",
        "python": sys.version.split()[0],
        "model": model,
        "embedding_files_sha256": {
            p.relative_to(embedding_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(embedding_path.rglob("*"))
            if p.is_file()
        },
        "adaptations": [
            "CPU all-MiniLM-L6-v2, 384 dimensions",
            "thinking disabled",
            "1024 output tokens per call, at most 12 calls",
            "non-streaming metered transport",
            "small synthetic history; no segmentation/compression ablations measured",
            "LightMem: pre_compress and topic_segment disabled, user_only, force_extract per batch",
            "SimpleMem: native text, reflection rounds <= 1, parallel processing disabled",
        ],
        "calls": transport.records,
    }
    started = time.monotonic()
    try:
        with (
            contextlib.redirect_stdout(captured),
            contextlib.redirect_stderr(captured),
            httpx.Client(
                transport=transport,
                timeout=30,
                follow_redirects=False,
            ) as http_client,
        ):
            client = OpenAI(
                api_key=PLACEHOLDER,
                base_url="https://api.deepseek.com",
                http_client=http_client,
                max_retries=0,
            )
            if reference == "simplemem":
                sys.path.insert(0, str(repo))
                from simplemem.core.settings import settings

                settings.EMBEDDING_MODEL = str(embedding_path)
                settings.EMBEDDING_DIMENSION = 384
                settings.ENABLE_PARALLEL_PROCESSING = False
                settings.ENABLE_PARALLEL_RETRIEVAL = False
                spec = importlib.util.spec_from_file_location(
                    "native_simplemem_main", repo / "main.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                native = module.SimpleMemSystem(
                    api_key=PLACEHOLDER,
                    model=model,
                    base_url="https://api.deepseek.com",
                    db_path=str(run_dir / "lance"),
                    use_streaming=False,
                    enable_parallel_processing=False,
                    enable_parallel_retrieval=False,
                    max_reflection_rounds=1,
                )
                native.llm_client.client.close()
                native.llm_client.client = client
                for date, content in HISTORY:
                    native.add_dialogue("user", content, date)
                native.finalize()
                memories = native.get_all_memories()
                context = native.hybrid_retriever.retrieve(QUERY)
                context = [
                    {"memory": m.lossless_restatement, "timestamp": m.timestamp} for m in context
                ]
                report["memory_entries"] = len(memories)
            else:
                from lightmem.memory.lightmem import LightMemory
                from qdrant_client import QdrantClient

                native = LightMemory.from_config(
                    {
                        "pre_compress": False,
                        "topic_segment": False,
                        "update": "offline",
                        "metadata_generate": True,
                        "text_summary": True,
                        "messages_use": "user_only",
                        "memory_manager": {
                            "model_name": "deepseek",
                            "configs": {
                                "model": model,
                                "api_key": PLACEHOLDER,
                                "deepseek_base_url": "https://api.deepseek.com",
                                "max_tokens": 1024,
                                "thinking": False,
                            },
                        },
                        "index_strategy": "embedding",
                        "retrieve_strategy": "embedding",
                        "text_embedder": {
                            "model_name": "huggingface",
                            "configs": {
                                "model": str(embedding_path),
                                "embedding_dims": 384,
                                "model_kwargs": {"device": "cpu"},
                            },
                        },
                        "embedding_retriever": {
                            "model_name": "qdrant",
                            "configs": {
                                "client": QdrantClient(":memory:"),
                                "path": str(run_dir / "qdrant-unused"),
                                "collection_name": "synthetic",
                                "embedding_model_dims": 384,
                                "on_disk": True,
                            },
                        },
                    }
                )
                native.manager.client.close()
                native.manager.client = client
                for date, content in HISTORY:
                    native.add_memory(
                        {"role": "user", "content": content, "time_stamp": date}, force_extract=True
                    )
                context = native.retrieve(QUERY, limit=5)
                report["native_statistics"] = native.get_token_statistics()
            report["retrieved_context"] = context
            raw = client.chat.completions.create(
                model=model,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Use only the supplied dated memory. Use the latest explicit employer. "
                            "Return JSON with name, employer, drink; unknown values must be null."
                        ),
                    },
                    {"role": "user", "content": json.dumps({"question": QUERY, "memory": context})},
                ],
            )
            answer = json.loads(raw.choices[0].message.content)
            report["answer"] = answer
            report["checks"] = {
                "name": str(answer.get("name", "")).lower() == "mira",
                "employer": "juniper" in str(answer.get("employer", "")).lower(),
                "drink": all(
                    v in str(answer.get("drink", "")).lower() for v in ("unsweetened", "oolong")
                ),
            }
            report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as error:
        report["error_type"] = type(error).__name__
        # Third-party errors may contain config or provider bodies. Keep them out of output.
    finally:
        report["seconds"] = round(time.monotonic() - started, 2)
        report["attempted_calls"] = len(transport.records)
        report["reported_tokens"] = sum(
            (v.get("usage") or {}).get("total_tokens", 0) for v in transport.records
        )
        report["usage_complete"] = all(v.get("usage") is not None for v in transport.records)
        sanitized_log = captured.getvalue().replace(secret, "[REDACTED]")
        (run_dir / "native.log").write_text(sanitized_log, encoding="utf-8")
        report["native_log"] = str((run_dir / "native.log").relative_to(ROOT))
        if secret in json.dumps(report):
            raise RuntimeError("Credential output rejected")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "reference": reference,
                "status": report["status"],
                "calls": report["attempted_calls"],
                "tokens": report["reported_tokens"],
            }
        )
    )
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", choices=REVISIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.reference, args.output) else 1)
