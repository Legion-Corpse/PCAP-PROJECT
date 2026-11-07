"""Publication must reject incomplete, unverified, or changed C17 evidence."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from c17_evidence import (configurations, digest, source_tree_hash, summarize_rows,
                          validate_artifacts, validate_manifest, validate_rows,
                          check_build_provenance, run_trials)


@pytest.fixture
def evidence(tmp_path):
    configs = configurations(ROOT)
    rows, commands = [], []
    for config in configs:
        for trial in range(5):
            commands.append(dict(configuration=config["name"], trial=trial,
                                 args=config["args"], process_wall_ms=10.0))
            for repeat in [0, 1]:
                rows.append(dict(implementation="c17", method=config["method"],
                    n_points=config["n_points"], n_queries=config["n_points"],
                    repeat=repeat, radius=config.get("radius", 0.01), group=8,
                    max_neighbors=0, index_ms=1.0, search_ms=2.0 + trial,
                    phase_total_ms=3.0 + trial, tiles=2, loads=2,
                    executor_bytes=1024, host_tile_bytes=2048, neighbors_total=10,
                    verify="sample", verify_checked=512,
                    verify_count_mismatches=0, verify_list_errors=0, status="ok",
                    configuration=config["name"], trial=trial, process_wall_ms=10.0))
    manifest = dict(schema_version=1, evidence_type="c17-independent-process",
        configurations=configs, commands=commands, trials_per_configuration=5,
        repeat_ids=[0, 1], measured_repeat=1, verify="sample", verify_samples=512,
        binary_sha256="a" * 64, source_tree_sha256="b" * 64,
        build_provenance=dict(status="built", binary_sha256="a" * 64,
            source_tree_sha256="b" * 64, configuration="Release", compiler="MSVC",
            cuda="13", optix_revision="9.1"))
    folder = tmp_path / "evidence"
    folder.mkdir()
    with zipfile.ZipFile(folder / "source.zip", "w") as archive:
        archive.writestr("src_c/example.c", "int x;")
    (tmp_path / "src_c").mkdir()
    (tmp_path / "src_c" / "example.c").write_text("int x;")
    manifest["source_tree_sha256"] = source_tree_hash(tmp_path)
    manifest["build_provenance"]["source_tree_sha256"] = manifest["source_tree_sha256"]
    manifest["source_sha256"] = digest(folder / "source.zip")
    (folder / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest["trials_sha256"] = digest(folder / "trials.jsonl")
    (folder / "summary.json").write_text(json.dumps(summarize_rows(rows, configs)))
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder, manifest, rows


def test_valid_complete_evidence_and_summary(evidence):
    folder, manifest, rows = evidence
    result = validate_manifest(folder / "manifest.json")
    assert len(result["rows"]) == 250
    assert result["summary"][0]["search_ms"] == dict(median=4.0, min=2.0, max=6.0)
    assert result["summary"][0]["phase_total_ms"]["median"] == 5.0


@pytest.mark.parametrize("key,value", [
    ("search_ms", -1), ("index_ms", float("nan")),
    ("phase_total_ms", float("inf")), ("verify_checked", 0),
    ("verify_count_mismatches", 1), ("verify_list_errors", 1),
    ("verify", "none"), ("status", "failed"),
    ("implementation", "cpp17"), ("method", "brute"),
    ("trial", 5), ("repeat", 2), ("n_points", 0),
    ("phase_total_ms", 99), ("process_wall_ms", float("nan")),
])
def test_invalid_row_rejected(evidence, key, value):
    _, manifest, rows = evidence
    rows[0][key] = value
    with pytest.raises(ValueError, match=key):
        validate_rows(rows, manifest)


@pytest.mark.parametrize("key", ["index_ms", "verify_checked", "verify_list_errors", "executor_bytes"])
def test_missing_required_field_is_not_measured_zero(evidence, key):
    _, manifest, rows = evidence
    del rows[0][key]
    with pytest.raises(ValueError, match=key):
        validate_rows(rows, manifest)


def test_missing_duplicate_trial_and_missing_configuration_rejected(evidence):
    _, manifest, rows = evidence
    for bad in [rows[:-1], rows + [rows[0]], rows[10:]]:
        with pytest.raises(ValueError, match="missing|duplicate|incomplete"):
            validate_rows(bad, manifest)


def test_omitted_configuration_in_manifest_rejected(evidence):
    _, manifest, rows = evidence
    manifest["configurations"].pop()
    with pytest.raises(ValueError, match="configuration"):
        validate_rows(rows, manifest)


def test_full_verification_requires_all_queries(evidence):
    _, manifest, rows = evidence
    manifest["verify"] = "full"
    for config in manifest["configurations"]:
        config["args"][config["args"].index("--verify") + 1] = "full"
    for row in rows:
        row["verify"] = "full"
        row["verify_checked"] = row["n_queries"]
    validate_rows(rows, manifest)
    rows[0]["verify_checked"] = 512
    with pytest.raises(ValueError, match="verify_checked"):
        validate_rows(rows, manifest)


@pytest.mark.parametrize("filename", ["trials.jsonl", "source.zip"])
def test_changed_archive_hash_rejected(evidence, filename):
    folder, _, _ = evidence
    with (folder / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=filename):
        validate_manifest(folder / "manifest.json")


def test_stale_summary_rejected(evidence):
    folder, _, _ = evidence
    path = folder / "summary.json"
    summary = json.loads(path.read_text())
    summary[0]["search_ms"]["median"] += 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary"):
        validate_manifest(folder / "manifest.json")


def test_artifact_hash_and_path_escape_rejected(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"pdf")
    hashes = tmp_path / "hashes.json"
    hashes.write_text(json.dumps({"report.pdf": digest(artifact)}))
    assert validate_artifacts(hashes, tmp_path) == 1
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="report.pdf"):
        validate_artifacts(hashes, tmp_path)
    hashes.write_text(json.dumps({"../outside.pdf": "a" * 64}))
    with pytest.raises(ValueError, match="outside"):
        validate_artifacts(hashes, tmp_path)


def test_build_record_rejects_changed_executable_and_source(tmp_path):
    (tmp_path / "src_c").mkdir()
    (tmp_path / "src_c" / "x.c").write_text("int x;")
    executable = tmp_path / "app.exe"
    executable.write_bytes(b"binary")
    record = dict(status="built", binary_sha256=digest(executable),
        source_tree_sha256=source_tree_hash(tmp_path), configuration="Release",
        compiler="MSVC", cuda="13", optix_revision="9.1")
    check_build_provenance(record, executable, tmp_path)
    executable.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary"):
        check_build_provenance(record, executable, tmp_path)
    executable.write_bytes(b"binary")
    (tmp_path / "src_c" / "x.c").write_text("int y;")
    with pytest.raises(ValueError, match="source"):
        check_build_provenance(record, executable, tmp_path)


def test_missing_optional_first_answer_is_null(evidence):
    _, manifest, rows = evidence
    summary = summarize_rows(rows, manifest["configurations"])
    assert summary[0]["first_answer_wall_ms"] is None
    assert summary[0]["executor_bytes"] is None
    assert summary[0]["host_tile_bytes"] is None
    assert summary[3]["executor_bytes"]["median"] == 1024


def test_runner_refuses_existing_output_before_launching(tmp_path):
    with pytest.raises(FileExistsError, match="overwrite"):
        run_trials(tmp_path / "unused.exe", tmp_path, {})


def test_manifest_cannot_relabel_snapshot_as_different_source(evidence):
    folder, manifest, _ = evidence
    manifest["source_tree_sha256"] = "c" * 64
    manifest["build_provenance"]["source_tree_sha256"] = "c" * 64
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source.zip tree"):
        validate_manifest(folder / "manifest.json")


def test_cold_wall_time_summary_uses_repeat_zero(evidence):
    _, manifest, rows = evidence
    for row in rows:
        row["first_answer_wall_ms"] = 100 + row["trial"] if row["repeat"] == 0 else None
    validate_rows(rows, manifest)
    summary = summarize_rows(rows, manifest["configurations"])
    assert summary[0]["first_answer_wall_ms"] == dict(median=102, min=100, max=104)
