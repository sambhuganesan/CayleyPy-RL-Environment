"""
generate_expected_hashes.py
----------------------------
Run ONCE during development to capture ground-truth hashes of the immutable files.
Output goes to scoring_data/cayleypy_review/expected_hashes.json (hidden from agent).

Run from the repo root:
    python generate_expected_hashes.py
"""

import hashlib
import json
from pathlib import Path


IMMUTABLE_FILES = [
    "paper/cayleypy.pdf",
    "generators/p054.json",
    "targets/p054-t000.pt",
    "datasets/p054-t000-deepcubeahard.pt",
]

# Adjust these two paths to match your actual layout.
# workdir  = the env_data directory where the immutable files live
# out_dir  = the scoring_data directory hidden from the agent
WORKDIR = Path("env_data/cayleypy_review")
OUT_DIR = Path("scoring_data/cayleypy_review")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for rel in IMMUTABLE_FILES:
        full_path = WORKDIR / rel
        if not full_path.exists():
            raise FileNotFoundError(f"Expected immutable file not found: {full_path}")
        hashes[rel] = file_sha256(full_path)
        print(f"  {rel}: {hashes[rel]}")

    out_path = OUT_DIR / "expected_hashes.json"
    out_path.write_text(json.dumps(hashes, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()