"""Repeated-batch evidence must use one index and preserve timing scope."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from c17_lifecycle import configurations, summarize_rows, validate_rows, validate_manifest
from c17_evidence import digest, source_tree_hash


@pytest.fixture
def evidence():
    configs = configurations()
    manifest = dict(schema_version=1, evidence_type="c17-reused-index-batches",
        configurations=configs, trials_per_configuration=5, repeat_ids=list(range(17)),
        prefix_batches=[1, 2, 4, 8, 16], commands=[])
    rows = []
    for config in configs:
        for trial in range(5):
            manifest["commands"].append(dict(configuration=config["name"], trial=trial,
                args=config["args"], process_wall_ms=100.0))
            for repeat in range(17):
                rows.append(dict(configuration=config["name"], trial=trial, repeat=repeat,
                    implementation="c17", method=config["method"],
                    n_points=config["n_points"], n_queries=config["n_points"],
                    group=8, radius=.01, max_neighbors=0, index_ms=10.0,
                    search_ms=50.0 if repeat == 0 else 2.0 + trial,
                    phase_total_ms=60.0 if repeat == 0 else 12.0 + trial,
                    status="ok", verify="sample", verify_checked=512,
                    verify_count_mismatches=0, verify_list_errors=0,
                    process_wall_ms=100.0))
    return rows, manifest


def test_prefix_sums_include_build_only_once_and_keep_cold_batch(evidence):
    rows, manifest = evidence
    validate_rows(rows, manifest)
    summary = summarize_rows(rows, manifest["configurations"])
    prefixes = summary[0]["prefixes"]
    assert [p["batches"] for p in prefixes] == [1, 2, 4, 8, 16]
    assert prefixes[0]["cold_phase_ms"]["median"] == 60.0
    assert prefixes[0]["warm_phase_ms"]["median"] == 14.0
    assert prefixes[-1]["cold_phase_ms"]["median"] == 10 + 50 + 15 * 4
    assert prefixes[-1]["warm_phase_ms"]["median"] == 10 + 16 * 4


@pytest.mark.parametrize("key,value", [("search_ms", float("nan")), ("index_ms", -1),
    ("verify_checked", 0), ("verify_count_mismatches", 1), ("verify_list_errors", 1),
    ("repeat", 17), ("trial", 5), ("verify", "none"), ("implementation", "cpp17"),
    ("process_wall_ms", float("inf")), ("phase_total_ms", 99)])
def test_corrupt_lifecycle_row_rejected(evidence, key, value):
    rows, manifest = evidence
    rows[0][key] = value
    with pytest.raises(ValueError, match=key):
        validate_rows(rows, manifest)


def test_build_timing_must_be_constant_within_one_process(evidence):
    rows, manifest = evidence
    rows[1]["index_ms"] = 11.0
    rows[1]["phase_total_ms"] += 1
    with pytest.raises(ValueError, match="index_ms.*index lifetime"):
        validate_rows(rows, manifest)


def test_missing_duplicate_repeats_and_missing_commands_rejected(evidence):
    rows, manifest = evidence
    for corrupted in [rows[:-1], rows + [rows[0]], rows[17:]]:
        with pytest.raises(ValueError, match="missing|duplicate|incomplete"):
            validate_rows(corrupted, manifest)
    manifest["commands"].pop()
    with pytest.raises(ValueError, match="command"):
        validate_rows(rows, manifest)


def test_summary_uses_median_of_trial_sums_not_sum_of_medians(evidence):
    rows, manifest = evidence
    name = manifest["configurations"][0]["name"]
    for row in rows:
        if row["configuration"] == name:
            row["index_ms"] = [0, 100, 100, 100, 0][row["trial"]]
            if row["repeat"] == 1:
                row["search_ms"] = [100, 0, 100, 0, 100][row["trial"]]
    prefix = summarize_rows(rows, manifest["configurations"])[0]["prefixes"][0]
    assert prefix["warm_phase_ms"]["median"] == 100
    assert prefix["warm_phase_ms"]["median"] != 200


def test_offline_validator_checks_snapshot_rows_and_derived_prefixes(evidence, tmp_path):
    rows, manifest = evidence
    (tmp_path / "src_c").mkdir()
    (tmp_path / "src_c" / "fixture.c").write_text("int fixture;")
    folder = tmp_path / "results"
    folder.mkdir()
    with zipfile.ZipFile(folder / "source.zip", "w") as archive:
        archive.write(tmp_path / "src_c" / "fixture.c", "src_c/fixture.c")
    (folder / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest.update(binary_sha256="a" * 64, source_tree_sha256=source_tree_hash(tmp_path),
        source_sha256=digest(folder / "source.zip"), trials_sha256=digest(folder / "trials.jsonl"))
    manifest["build_provenance"] = dict(status="built", binary_sha256=manifest["binary_sha256"],
        source_tree_sha256=manifest["source_tree_sha256"], configuration="Release",
        compiler="MSVC", cuda="13", optix_revision="9.1")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    summary = summarize_rows(rows, manifest["configurations"])
    (folder / "summary.json").write_text(json.dumps(summary))
    assert len(validate_manifest(folder / "manifest.json")["rows"]) == 510
    summary[0]["prefixes"][0]["warm_phase_ms"]["median"] += 1
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary"):
        validate_manifest(folder / "manifest.json")
    with (folder / "trials.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="trials.jsonl"):
        validate_manifest(folder / "manifest.json")
