#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/journal.json}"
OUTPUT="${2:-results/journal}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$OUTPUT"

python scripts/run_from_config.py "$CONFIG" --output "$OUTPUT"

python --version > "$OUTPUT/python_version.txt" 2>&1
python -m pip freeze > "$OUTPUT/environment_freeze.txt"

python - "$OUTPUT" <<'PY'
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
manifest_path = output / "artifact_manifest.json"

environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
}
(output / "environment.json").write_text(
    json.dumps(environment, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

files = []
for path in sorted(p for p in output.rglob("*") if p.is_file()):
    if path == manifest_path:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append(
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    )

manifest = {
    "schema_version": 1,
    "output_root": output.name,
    "hash_algorithm": "sha256",
    "file_count": len(files),
    "files": files,
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

echo "Reproduction complete: $OUTPUT"
echo "Config snapshot: $OUTPUT/config_used.json"
echo "Environment freeze: $OUTPUT/environment_freeze.txt"
echo "Artifact manifest: $OUTPUT/artifact_manifest.json"
