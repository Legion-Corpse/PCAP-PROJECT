"""Fresh C17 trials and strict offline publication validation.

Run only after building the C17 executable and recording build provenance. Each
process reuses one point index for two query batches; repeat 0 is warm-up and
repeat 1 is the retained search. index+retained search is a derived phase sum,
not an independently measured cold first-answer wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("cpu-grid", "cuda-grid", "rt-incore", "rt-ooc")
METRICS = ("index_ms", "search_ms", "phase_total_ms", "process_wall_ms",
           "executor_bytes", "host_tile_bytes", "first_answer_wall_ms")
BUILD_FIELDS = ("binary_sha256", "source_tree_sha256", "configuration", "compiler",
                "cuda", "optix_revision")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(root):
    root = Path(root)
    return sorted([p for directory in ("src_c", "cmake", "examples")
                   for p in (root / directory).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts]
                  + ([root / "CMakeLists.txt"] if (root / "CMakeLists.txt").is_file() else []),
                  key=lambda p: p.relative_to(root).as_posix())


def _tree_hash(entries):
    """SHA256 of sorted path + NUL + raw file SHA256 + LF records."""
    result = hashlib.sha256()
    for name, content in sorted(entries):
        result.update(name.encode("utf-8") + b"\0" + hashlib.sha256(content).digest() + b"\n")
    return result.hexdigest()


def source_tree_hash(root=ROOT):
    root = Path(root)
    return _tree_hash((p.relative_to(root).as_posix(), p.read_bytes()) for p in source_files(root))


def _snapshot_tree_hash(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("source.zip has duplicate entries")
        return _tree_hash((name, archive.read(name)) for name in names
                          if name == "CMakeLists.txt" or
                          (name.startswith(("src_c/", "cmake/", "examples/")) and not name.endswith("/")))


def check_build_provenance(record, executable=None, root=ROOT):
    if record.get("status") != "built":
        raise ValueError("build provenance must have status=built")
    for field in BUILD_FIELDS:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"build provenance missing {field}")
    for field in ("binary_sha256", "source_tree_sha256"):
        if len(record[field]) != 64 or any(c not in "0123456789abcdef" for c in record[field]):
            raise ValueError(f"build provenance invalid {field}")
    if executable is not None:
        if digest(executable) != record["binary_sha256"]:
            raise ValueError("binary hash differs from build provenance; rebuild and record it")
        if source_tree_hash(root) != record["source_tree_sha256"]:
            raise ValueError("source tree differs from build provenance; rebuild and record it")


def configurations(root=ROOT):
    root = Path(root)
    datasets = [("uniform100K", "uniform:100K", 100_000, None),
                ("uniform1M", "uniform:1M", 1_000_000, None),
                ("uniform4M", "uniform:4M", 4_000_000, None),
                ("scene200K", "scene:200K", 200_000, .015),
                ("armadillo", "file:" + str(root / "data/stanford/Armadillo.ply"), 172_974, 1.0)]
    configs = []
    for label, points, count, radius in datasets:
        for method in METHODS:
            variants = [("64K", "on")]
            if label == "uniform1M" and method == "rt-ooc":
                variants += [("16K", "on"), ("256K", "on"), ("16K", "off"),
                             ("64K", "off"), ("0", "on")]
            for tile, overlap in variants:
                name = f"{label}_{method}"
                if (tile, overlap) != ("64K", "on"):
                    name += "_auto" if tile == "0" else f"_tile{tile}_overlap{overlap}"
                args = ["--points", points, "--queries", "same", "--method", method,
                        "--group", "8", "--budget-mb", "64", "--tile-points", tile,
                        "--slots", "2", "--overlap", overlap, "--repeat", "2",
                        "--verify", "sample", "--verify-samples", "512", "--seed", "1"]
                args += ["--k-expected", "32"] if radius is None else ["--radius", str(radius)]
                config = dict(name=name, method=method, n_points=count, points=points,
                              tile_points=tile, overlap=overlap, args=args)
                if radius is not None:
                    config["radius"] = radius
                configs.append(config)
    return configs


def _number(value, field, where, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {field} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{where}: {field} must be finite and {'positive' if positive else 'nonnegative'}")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{where}: {field} must be an integer")
    return value


def _catalog(manifest):
    configs = manifest.get("configurations", [])
    expected = {c["name"]: c for c in configurations()}
    if len(configs) != len(expected) or {c.get("name") for c in configs} != set(expected):
        raise ValueError("configuration catalog incomplete or unexpected (expected 25 configurations)")
    for config in configs:
        reference = expected[config["name"]]
        for key in ("method", "n_points", "tile_points", "overlap", "radius"):
            if config.get(key) != reference.get(key):
                raise ValueError(f"configuration {config['name']}: incorrect {key}")
        args = config.get("args", [])
        if len(args) % 2 or len(set(args[::2])) != len(args) // 2:
            raise ValueError(f"configuration {config['name']}: invalid args")
        actual, wanted = dict(zip(args[::2], args[1::2])), dict(zip(reference["args"][::2], reference["args"][1::2]))
        wanted["--verify"] = manifest.get("verify")
        # Preserve archived file paths across machines; all other options are fixed.
        if reference["points"].startswith("file:"):
            if not actual.get("--points", "").startswith("file:"):
                raise ValueError(f"configuration {config['name']}: missing file input")
            wanted["--points"] = actual["--points"]
        if actual != wanted:
            raise ValueError(f"configuration {config['name']}: command args differ from protocol")
    return {c["name"]: c for c in configs}


def validate_rows(rows, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("evidence_type") != "c17-independent-process":
        raise ValueError("unsupported C17 evidence schema_version or evidence_type")
    for field, value in dict(trials_per_configuration=5, repeat_ids=[0, 1], measured_repeat=1,
                             verify_samples=512).items():
        if manifest.get(field) != value:
            raise ValueError(f"manifest {field} differs from protocol")
    if manifest.get("verify") not in ("sample", "full"):
        raise ValueError("manifest verify must be sample or full")
    configs = _catalog(manifest)
    command_map = {}
    for command in manifest.get("commands", []):
        name, trial = command.get("configuration"), command.get("trial")
        _number(trial, "trial", "command", integer=True)
        if name not in configs or trial not in range(5):
            raise ValueError("command configuration/trial outside protocol")
        key = (name, trial)
        if key in command_map:
            raise ValueError(f"duplicate command {key}")
        if command.get("args") != configs[name]["args"]:
            raise ValueError(f"command args differ from configuration {name}")
        _number(command.get("process_wall_ms"), "process_wall_ms", str(key), positive=True)
        command_map[key] = command
    expected_commands = {(name, trial) for name in configs for trial in range(5)}
    if set(command_map) != expected_commands:
        raise ValueError("missing/incomplete command trials")
    seen = set()
    for row in rows:
        where = f"{row.get('configuration')} trial={row.get('trial')} repeat={row.get('repeat')}"
        for field in ("configuration", "trial", "repeat", "implementation", "method", "n_points",
                      "n_queries", "radius", "group", "max_neighbors", "index_ms", "search_ms",
                      "phase_total_ms", "tiles", "loads", "executor_bytes", "host_tile_bytes",
                      "neighbors_total", "verify", "verify_checked", "verify_count_mismatches",
                      "verify_list_errors", "status", "process_wall_ms"):
            if field not in row:
                raise ValueError(f"{where}: missing required {field}")
        name, trial, repeat = row["configuration"], row["trial"], row["repeat"]
        for field in ("trial", "repeat"):
            _number(row[field], field, where, integer=True)
        if name not in configs:
            raise ValueError(f"{where}: unexpected configuration")
        if trial not in range(5) or repeat not in (0, 1):
            raise ValueError(f"{where}: trial/repeat outside protocol")
        key = (name, trial, repeat)
        if key in seen:
            raise ValueError(f"{where}: duplicate row")
        seen.add(key)
        config = configs[name]
        for field, value in (("implementation", "c17"), ("method", config["method"]),
                             ("n_points", config["n_points"]), ("n_queries", config["n_points"]),
                             ("group", 8), ("max_neighbors", 0), ("status", "ok"), ("verify", manifest["verify"]),
                             ("verify_checked", 512 if manifest["verify"] == "sample" else config["n_points"]),
                             ("verify_count_mismatches", 0), ("verify_list_errors", 0)):
            if row[field] != value or isinstance(row[field], bool):
                raise ValueError(f"{where}: {field} must equal {value!r}")
        _number(row["radius"], "radius", where, positive=True)
        if "radius" in config and not math.isclose(row["radius"], config["radius"], rel_tol=1e-6):
            raise ValueError(f"{where}: radius differs from command")
        for field, value in row.items():
            if field.endswith("_ms"):
                if field == "first_answer_wall_ms" and value is None and repeat > 0:
                    continue
                _number(value, field, where)
        for field in ("n_points", "n_queries", "tiles", "loads", "executor_bytes", "host_tile_bytes",
                      "neighbors_total", "verify_checked", "verify_count_mismatches", "verify_list_errors"):
            _number(row[field], field, where, integer=True)
        if not math.isclose(row["phase_total_ms"], row["index_ms"] + row["search_ms"], abs_tol=2.1e-6, rel_tol=1e-8):
            raise ValueError(f"{where}: phase_total_ms differs from index_ms + search_ms")
        if row["process_wall_ms"] != command_map[(name, trial)]["process_wall_ms"]:
            raise ValueError(f"{where}: process_wall_ms differs from command")
    expected_rows = {(name, trial, repeat) for name in configs for trial in range(5) for repeat in (0, 1)}
    if seen != expected_rows:
        raise ValueError(f"missing/incomplete trial repetitions: {len(expected_rows-seen)} rows missing")
    return rows


def summarize_rows(rows, configs):
    summary = []
    for config in configs:
        selected = [r for r in rows if r["configuration"] == config["name"] and r["repeat"] == 1]
        if len(selected) != 5:
            raise ValueError(f"incomplete measured trials for {config['name']}")
        entry = dict(configuration=config["name"], method=config["method"], n=5)
        for metric in METRICS:
            if metric in ("executor_bytes", "host_tile_bytes") and config["method"] not in ("rt-incore", "rt-ooc"):
                entry[metric] = None
                continue
            metric_rows = selected
            if metric == "first_answer_wall_ms":
                metric_rows = [r for r in rows if r["configuration"] == config["name"] and r["repeat"] == 0]
            values = [r.get(metric) for r in metric_rows]
            if all(v is None for v in values):
                entry[metric] = None
            elif any(v is None for v in values):
                raise ValueError(f"partial metric {metric} for {config['name']}")
            else:
                for value in values:
                    _number(value, metric, config["name"])
                entry[metric] = dict(median=statistics.median(values), min=min(values), max=max(values))
        summary.append(entry)
    return summary


def validate_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    folder = path.parent
    for name, field in (("source.zip", "source_sha256"), ("trials.jsonl", "trials_sha256")):
        if not (folder / name).is_file() or digest(folder / name) != manifest.get(field):
            raise ValueError(f"{name}: missing file or SHA256 mismatch")
    record = manifest.get("build_provenance", {})
    check_build_provenance(record)
    for field in ("binary_sha256", "source_tree_sha256"):
        if manifest.get(field) != record[field]:
            raise ValueError(f"manifest {field} differs from build provenance")
    if _snapshot_tree_hash(folder / "source.zip") != record["source_tree_sha256"]:
        raise ValueError("source.zip tree hash differs from build provenance")
    rows = [json.loads(line) for line in (folder / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_rows(rows, manifest)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary != summarize_rows(rows, manifest["configurations"]):
        raise ValueError("summary.json differs from recomputed medians/min/max")
    return dict(manifest=manifest, rows=rows, summary=summary)


def validate_artifacts(path, root=ROOT):
    root = Path(root).resolve()
    hashes = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("artifact hash manifest must be a nonempty mapping")
    for name, expected in hashes.items():
        target = (root / name.replace("\\", "/")).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"artifact outside project root: {name}")
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f"artifact {name}: missing file or SHA256 mismatch")
    return len(hashes)


def run_trials(executable, output, provenance):
    executable, output = Path(executable).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence directory: {output}")
    check_build_provenance(provenance, executable)
    real = ROOT / "data/stanford/Armadillo.ply"
    if not real.is_file():
        raise FileNotFoundError(f"Prepare the recorded Armadillo input first: {real}")
    configs = configurations()
    manifest = dict(schema_version=1, evidence_type="c17-independent-process",
        created_utc=datetime.now(timezone.utc).isoformat(), random_seed=20261029,
        configurations=configs, commands=[], trials_per_configuration=5,
        repeat_ids=[0, 1], measured_repeat=1, verify="sample", verify_samples=512,
        binary_sha256=digest(executable), source_tree_sha256=source_tree_hash(),
        build_provenance=provenance, binary_path=str(executable),
        dataset_sha256={"Armadillo.ply": digest(real)},
        environment={"platform": platform.platform(), "python": platform.python_version(),
                     "gpu": subprocess.check_output(["nvidia-smi"], text=True, timeout=30)},
        metric_scopes={"executor_bytes": "RT backend explicit executor allocations only; unavailable for CPU/CUDA grid",
                       "host_tile_bytes": "RT backend host tile store only; unavailable for CPU/CUDA grid",
                       "unavailable_summary_metrics": "null; raw CLI zero placeholders do not establish memory usage"},
        timing_contract="index_ms: build once before repeat loop; search_ms: query batch against reused index; repeat0 warmup excluded from search summary; phase_total_ms: derived build+retained search; process_wall_ms: complete process including both batches, verification and startup; first_answer_wall_ms: separately measured repeat0 construction-start to first-query completion before verification, excludes input generation and process startup")
    output.mkdir(parents=True)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in source_files(ROOT):
            archive.write(source, source.relative_to(ROOT).as_posix())
        archive.write(Path(__file__), "bench/c17_evidence.py")
    manifest["source_sha256"] = digest(output / "source.zip")
    jobs = [(config, trial) for config in configs for trial in range(5)]
    random.Random(manifest["random_seed"]).shuffle(jobs)
    rows = []
    partial = output / "trials.partial.jsonl"
    partial.write_text("", encoding="utf-8")
    for i, (config, trial) in enumerate(jobs):
        check_build_provenance(provenance, executable)
        path = output / "command.partial.jsonl"
        path.write_text("", encoding="utf-8")
        command = [str(executable), *config["args"], "--json", str(path)]
        print(f"{i+1}/{len(jobs)} {config['name']} trial={trial}", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        wall = (time.perf_counter() - start) * 1000
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
        emitted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(emitted) != 2 or sorted(r.get("repeat", -1) for r in emitted) != [0, 1]:
            raise ValueError(f"{config['name']}: missing/duplicate repeat rows")
        for row in emitted:
            row.update(configuration=config["name"], trial=trial, process_wall_ms=wall)
        rows.extend(emitted)
        with partial.open("a", encoding="utf-8") as stream:
            for row in emitted:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        manifest["commands"].append(dict(configuration=config["name"], trial=trial,
            args=config["args"], process_wall_ms=wall))
        (output / "manifest.partial.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check_build_provenance(provenance, executable)
    if digest(real) != manifest["dataset_sha256"]["Armadillo.ply"]:
        raise ValueError("Armadillo input changed during benchmark")
    validate_rows(rows, manifest)
    # Final publication marker is written only after validating all data and summaries.
    partial.replace(output / "trials.jsonl")
    manifest["trials_sha256"] = digest(output / "trials.jsonl")
    (output / "summary.json").write_text(json.dumps(summarize_rows(rows, configs), indent=2, allow_nan=False), encoding="utf-8")
    candidate = output / "manifest.partial.json"
    candidate.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(candidate)
    candidate.replace(output / "manifest.json")
    print(f"Published {len(rows)} validated C17 rows in {output}", flush=True)
    return validate_manifest(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, metavar="MANIFEST")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--binary", type=Path, default=os.environ.get("OOC_RTNN", str(ROOT / "build_c17/ooc_rtnn.exe")))
    parser.add_argument("--build-record", type=Path, default=ROOT / "build_c17/build_provenance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/c17_20261029")
    args = parser.parse_args()
    try:
        if args.run:
            run_trials(args.binary, args.output, json.loads(args.build_record.read_text(encoding="utf-8-sig")))
        elif args.validate:
            result = validate_manifest(args.validate)
            print(f"C17 evidence: {len(result['rows'])} rows, {len(result['summary'])} complete configurations, hashes/verification/summaries valid")
        elif not args.artifacts:
            parser.error("select --run, --validate MANIFEST, or --artifacts HASHES")
        if args.artifacts:
            print(f"Artifacts: {validate_artifacts(args.artifacts)} SHA256 hashes valid")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Evidence validation failed: {error}\n")


if __name__ == "__main__":
    main()
