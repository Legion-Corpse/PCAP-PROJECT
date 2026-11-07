"""Behavioural checks for the ooc_rtnn command-line executable."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("OOC_RTNN", ROOT / "build_c17" / "ooc_rtnn.exe"))
pytestmark = pytest.mark.skipif(not BIN.is_file(), reason="C17 executable is not built")


def run(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    output = tmp_path / "c17.jsonl"
    process = subprocess.run([str(BIN), *args, "--json", str(output)],
                             capture_output=True, text=True, timeout=600)
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    return process, rows


def assert_verified(process: subprocess.CompletedProcess[str], rows: list[dict], count: int) -> None:
    assert process.returncode == 0, process.stdout + process.stderr
    assert len(rows) == count
    for row in rows:
        assert row["implementation"] == "c17", "test selected an archived C++ executable"
        assert row["status"] == "ok" and row["verify"] == "full"
        assert row["verify_checked"] == row["n_queries"] > 0
        assert row["verify_count_mismatches"] == row["verify_list_errors"] == 0
        for metric in ("index_ms", "search_ms", "phase_total_ms"):
            assert math.isfinite(row[metric]) and row[metric] >= 0


def test_c17_help():
    process = subprocess.run([str(BIN), "--help"], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0
    assert "C17" in process.stdout and "--method" in process.stdout
    assert "MiB" in process.stdout and "executor" in process.stdout


@pytest.mark.parametrize("points", ["uniform:40K", "clustered:40K:5", "scene:40K"])
def test_c17_all_gpu_methods_agree_with_cpu_grid(tmp_path, points):
    process, rows = run(tmp_path, "--points", points, "--method",
                        "cpu-grid,cuda-grid,rt-incore,rt-ooc", "--verify", "full",
                        "--tile-points", "4K", "--budget-mb", "16", "--group", "8")
    assert_verified(process, rows, 4)
    assert {row["method"] for row in rows} == {"cpu-grid", "cuda-grid", "rt-incore", "rt-ooc"}
    assert len({row["neighbors_total"] for row in rows}) == 1
    cuda = next(row for row in rows if row["method"] == "cuda-grid")
    assert cuda["executor_bytes"] >= 20 * cuda["n_points"] + 20 * cuda["n_queries"]
    ooc = next(row for row in rows if row["method"] == "rt-ooc")
    assert ooc["tiles"] > 1 and ooc["loads"] >= ooc["tiles"]
    assert 0 < ooc["executor_bytes"] <= 16 * 1024 * 1024
    assert f"executor={ooc['executor_bytes'] / 1048576:.3f} MiB" in process.stdout
    assert f"pinned_tiles={ooc['host_tile_bytes'] / 1048576:.3f} MiB" in process.stdout


def test_c17_distinct_queries_and_bounded_lists(tmp_path):
    process, rows = run(tmp_path, "--points", "clustered:40K:5", "--queries", "uniform:8K",
                        "--radius", "0.1", "--max-neighbors", "3", "--group", "3",
                        "--method", "cpu-grid,cuda-grid,rt-incore,rt-ooc", "--verify", "full",
                        "--tile-points", "4K", "--budget-mb", "16", "--window", "511")
    assert_verified(process, rows, 4)
    assert all(row["n_queries"] == 8000 and row["max_neighbors"] == 3 for row in rows)
    assert len({row["neighbors_total"] for row in rows}) == 1
    # Full counts must include neighbors beyond the three stored identifiers.
    assert rows[0]["neighbors_total"] > 8000 * 3


@pytest.mark.parametrize("overlap,slots", [("on", "2"), ("off", "1")])
def test_c17_repeated_query_batches_reuse_index(tmp_path, overlap, slots):
    process, rows = run(tmp_path, "--points", "uniform:40K", "--queries", "scene:4K",
                        "--method", "cpu-grid,rt-ooc", "--verify", "full", "--repeat", "3",
                        "--tile-points", "4K", "--budget-mb", "16", "--slots", slots,
                        "--overlap", overlap)
    assert_verified(process, rows, 6)
    for method in ("cpu-grid", "rt-ooc"):
        group = [row for row in rows if row["method"] == method]
        assert [row["repeat"] for row in group] == [0, 1, 2]
        assert len({row["index_ms"] for row in group}) == 1
        assert len({row["neighbors_total"] for row in group}) == 1
    assert len({row["neighbors_total"] for row in rows}) == 1


def test_c17_tiny_executor_budget_fails_without_measurement(tmp_path):
    process, rows = run(tmp_path, "--points", "uniform:200K", "--method", "rt-ooc",
                        "--tile-points", "200K", "--group", "1", "--budget-mb", "1",
                        "--slots", "2", "--verify", "full")
    assert process.returncode != 0
    assert "budget" in (process.stdout + process.stderr).lower()
    assert not rows


@pytest.mark.parametrize("option,value", [
    ("--repeat", "0"), ("--repeat", "-1"), ("--repeat", "2147483648"),
    ("--repeat", "1oops"), ("--repeat", "nan"), ("--repeat", ""),
    ("--verify", "typo"), ("--verify-samples", "0"), ("--verify-samples", "-1"),
    ("--radius", "0"), ("--radius", "-1"), ("--radius", "nan"),
    ("--radius", "inf"), ("--radius", "1e100"), ("--radius", "0.1oops"),
    ("--radius", " 0.1"), ("--radius", "1e-9999"), ("--radius", "1e-30"), ("--radius", "1e-20"),
    ("--budget-mb", "0"), ("--budget-mb", "-1"), ("--budget-mb", "nan"),
    ("--budget-mb", "inf"), ("--budget-mb", "1e100"), ("--budget-mb", "1oops"),
    ("--max-neighbors", "-1"), ("--max-neighbors", "4294967296"),
    ("--group", "0"), ("--group", "4294967296"),
    ("--tile-points", "-1"), ("--tile-points", "4294967296"),
    ("--window", "-1"), ("--seed", "-1"), ("--slots", "0"),
    ("--slots", "3"), ("--slots", "1oops"), ("--overlap", "typo"),
    ("--method", "typo"), ("--method", ""), ("--k-expected", "0"),
])
def test_c17_invalid_option_preflight(tmp_path, option, value):
    missing = tmp_path / "missing.xyz"
    process, rows = run(tmp_path, "--points", f"file:{missing}", "--method", "brute", option, value)
    assert process.returncode != 0
    assert "error" in process.stderr.lower()
    assert "missing.xyz" not in process.stderr, "invalid options must be rejected before dataset I/O"
    assert not rows


def test_c17_unknown_option_and_missing_value_rejected(tmp_path):
    for args in (("--unknown-option", "3"), ("--radius",)):
        process = subprocess.run([str(BIN), "--points", "uniform:8", "--method", "brute", *args],
                                 capture_output=True, text=True, timeout=30)
        assert process.returncode != 0 and process.stderr


def test_c17_json_creation_failure_is_actionable(tmp_path):
    process = subprocess.run([str(BIN), "--points", f"file:{tmp_path / 'missing.xyz'}",
                              "--method", "brute", "--json", str(tmp_path)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert "JSON" in process.stderr and "missing.xyz" not in process.stderr


@pytest.mark.parametrize("contents", [b"", struct.pack("<fff", math.nan, 0, 0), b"bad"])
def test_c17_empty_nonfinite_or_truncated_input_rejected(tmp_path, contents):
    source = tmp_path / "invalid.bin"
    source.write_bytes(contents)
    process, rows = run(tmp_path, "--points", f"file:{source}", "--method", "brute", "--radius", "1")
    assert process.returncode != 0 and process.stderr
    assert not rows


def test_c17_cpu_brute_exact_boundaries_and_duplicate_points(tmp_path):
    source = tmp_path / "duplicates.xyz"
    source.write_text("0 0 0\n" * 17 + "1 0 0\n0 1 0\n")
    process, rows = run(tmp_path, "--points", f"file:{source}", "--radius", "1",
                        "--max-neighbors", "3", "--method", "brute,cpu-grid,cuda-grid,rt-incore,rt-ooc",
                        "--verify", "full", "--group", "3", "--tile-points", "5")
    assert_verified(process, rows, 5)
    assert {row["neighbors_total"] for row in rows} == {359}


def test_c17_suffix_sample_coverage_and_json_append(tmp_path):
    args = ("--points", "uniform:0.008K", "--method", "brute", "--verify", "sample",
            "--verify-samples", "3")
    process, rows = run(tmp_path, *args)
    assert process.returncode == 0, process.stdout + process.stderr
    assert len(rows) == 1 and rows[0]["implementation"] == "c17"
    assert rows[0]["n_points"] == 8 and rows[0]["verify_checked"] == 3
    process, rows = run(tmp_path, *args)
    assert process.returncode == 0 and len(rows) == 2
