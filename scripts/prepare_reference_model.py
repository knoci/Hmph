"""Prepare the pinned public CPU embedding model in the native reference environment."""

import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    target = ROOT / "artifacts/models/all-minilm-native"
    # Never overwrite an existing model with potentially different files.
    if target.exists():
        raise RuntimeError("Model directory already exists; use the recorded manifest")
    model = SentenceTransformer(
        MODEL,
        revision=REVISION,
        device="cpu",
        cache_folder=str(ROOT / "artifacts/models/sentence-transformers"),
    )
    model.save_pretrained(str(target))
    files = {
        p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
    (target.parent / "all-minilm-native-manifest.json").write_text(
        json.dumps({"model": MODEL, "revision": REVISION, "files": files}, indent=2),
        encoding="utf-8",
    )
    print("Pinned public CPU model prepared; files remain under ignored artifacts.")


if __name__ == "__main__":
    main()
