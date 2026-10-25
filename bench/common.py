"""Shared helpers: locating the binary, running it, loading JSON-lines results."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FIGURES = RESULTS / "figures"


def binary() -> Path:
    """Path of the ooc_rtnn executable (override with $OOC_RTNN)."""
    env = os.environ.get("OOC_RTNN")
    if env:
        return Path(env)
    for name in ("ooc_rtnn.exe", "ooc_rtnn"):
        for sub in ("build", "build/Release"):
            p = ROOT / sub / name
            if p.exists():
                return p
    raise FileNotFoundError("ooc_rtnn not built; run scripts/build.ps1 (or set $OOC_RTNN)")


def run(args: list[str], json_path: Path | None = None, timeout: float | None = None,
        echo: bool = True) -> subprocess.CompletedProcess:
    """Runs the CLI; with json_path, results are appended there as JSON lines."""
    cmd = [str(binary()), *map(str, args)]
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--json", str(json_path)]
    if echo:
        print("$", " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if echo:
        for line in proc.stdout.splitlines():
            if line.startswith(("[", "  run:", "  index:", "  ->")):
                print("   ", line)
    if proc.returncode not in (0, 1):  # 1 = verification mismatch, still has JSON
        sys.stderr.write(proc.stdout + proc.stderr)
    return proc


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_frame(path: Path):
    import pandas as pd

    return pd.DataFrame(load(path))
