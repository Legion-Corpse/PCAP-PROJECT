"""End-to-end tests of the ooc_rtnn binary: every backend, verified against the CPU reference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import binary  # noqa: E402

try:
    BIN = binary()
except FileNotFoundError:  # pragma: no cover - depends on the build
    BIN = None

pytestmark = pytest.mark.skipif(BIN is None, reason="ooc_rtnn is not built")


def run(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess, list[dict]]:
    out = tmp_path / "r.jsonl"
    proc = subprocess.run([str(BIN), *args, "--json", str(out)], capture_output=True, text=True, timeout=600)
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    return proc, rows


def test_help():
    proc = subprocess.run([str(BIN), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--method" in proc.stdout and "rt-ooc" in proc.stdout


@pytest.mark.parametrize("points", ["uniform:30K", "clustered:30K:4", "scene:30K"])
def test_all_methods_agree(tmp_path, points):
    proc, rows = run(tmp_path, "--points", points, "--k-expected", "20", "--method",
                     "brute,cpu-grid,cuda-grid,rt-incore,rt-ooc", "--verify", "full", "--budget-mb", "4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {r["method"] for r in rows} == {"brute", "cpu-grid", "cuda-grid", "rt-incore", "rt-ooc"}
    assert all(r["status"] == "ok" for r in rows)
    assert len({r["neighbors_total"] for r in rows}) == 1  # identical answers


def test_neighbor_lists_and_queries_set(tmp_path):
    proc, rows = run(tmp_path, "--points", "uniform:20K", "--queries", "clustered:8K:3", "--radius", "0.06",
                     "--max-neighbors", "12", "--method", "cuda-grid,rt-incore,rt-ooc", "--group", "4",
                     "--verify", "full")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert all(r["status"] == "ok" and r["n_queries"] == 8000 for r in rows)
    assert all(r["truncated_queries"] > 0 for r in rows)  # K below the max count: rows are truncated, counts exact


def test_ooc_variants_under_tiny_budget(tmp_path):
    proc, rows = run(tmp_path, "--points", "uniform:40K", "--k-expected", "16", "--method", "rt-ooc",
                     "--budget-mb", "2", "--leaf-points", "512", "--order", "tiled,morton,random,original",
                     "--eviction", "belady,lru", "--slots", "1,2", "--verify", "full")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(rows) == 4 * 2 * 2
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["tiles"] > 1 for r in rows)
    # Belady (offline optimal) never needs more tile loads than LRU for the same plan.
    by = {(r["order"], r["slots"], r["eviction"]): r["tile_loads"] for r in rows}
    for (order, slots, ev), loads in by.items():
        if ev == "belady":
            assert loads <= by[(order, slots, "lru")]


def test_halo_modes(tmp_path):
    proc, rows = run(tmp_path, "--points", "scene:40K", "--radius", "0.01", "--method", "rt-ooc", "--depth", "4",
                     "--halo", "queries", "--group", "3", "--verify", "full")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert rows[0]["halo"] == "queries" and rows[0]["pairs_per_query"] >= 1.0


def test_unknown_option_warns():
    proc = subprocess.run([str(BIN), "--points", "uniform:1K", "--method", "cpu-grid", "--verify", "none",
                           "--definitely-not-an-option", "3"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "unknown option --definitely-not-an-option" in proc.stderr
