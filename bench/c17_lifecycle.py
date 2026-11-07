"""Measured query batches on one reused index; phase sums, not process latency.

Every independent process builds once and executes 17 batches against identical
points and queries. Cold prefixes start at batch 0; warm prefixes exclude batch
0 and start at batch 1. Both phase sums include construction exactly once.
Verification and process startup are outside these phase sums.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import zipfile

from c17_evidence import (ROOT, _number, _snapshot_tree_hash, check_build_provenance,
                          digest, source_files, source_tree_hash)

PREFIXES = (1, 2, 4, 8, 16)


def configurations():
    configs = []
    for size, count in (("1M", 1_000_000), ("4M", 4_000_000)):
        for method in ("cuda-grid", "rt-incore", "rt-ooc"):
            args = ["--points", f"uniform:{size}", "--queries", "same", "--method", method,
                    "--group", "8", "--budget-mb", "64", "--tile-points", "64K",
                    "--slots", "2", "--overlap", "on", "--repeat", "17", "--verify", "sample",
                    "--verify-samples", "512", "--seed", "1", "--k-expected", "32"]
            configs.append(dict(name=f"uniform{size}_{method}", method=method,
                                n_points=count, args=args))
    return configs


def validate_rows(rows, manifest):
    for field, value in dict(schema_version=1, evidence_type="c17-reused-index-batches",
        trials_per_configuration=5, repeat_ids=list(range(17)), prefix_batches=list(PREFIXES),
        configurations=configurations()).items():
        if manifest.get(field) != value:
            raise ValueError(f"manifest {field} differs from lifecycle protocol")
    configs = {c["name"]: c for c in manifest["configurations"]}
    commands = {}
    for command in manifest.get("commands", []):
        name, trial = command.get("configuration"), command.get("trial")
        _number(trial, "trial", "command", integer=True)
        if name not in configs or trial not in range(5):
            raise ValueError("command configuration/trial outside lifecycle protocol")
        if (name, trial) in commands:
            raise ValueError("duplicate command")
        if command.get("args") != configs[name]["args"]:
            raise ValueError(f"command args differ for {name}")
        _number(command.get("process_wall_ms"), "process_wall_ms", name, positive=True)
        commands[(name, trial)] = command
    if set(commands) != {(name, trial) for name in configs for trial in range(5)}:
        raise ValueError("missing/incomplete command trials")
    seen, builds = set(), {}
    for row in rows:
        where = f"{row.get('configuration')} trial={row.get('trial')} repeat={row.get('repeat')}"
        required = ("configuration", "trial", "repeat", "implementation", "method", "n_points",
            "n_queries", "radius", "group", "max_neighbors", "index_ms", "search_ms", "phase_total_ms",
            "status", "verify", "verify_checked", "verify_count_mismatches", "verify_list_errors",
            "process_wall_ms")
        for field in required:
            if field not in row:
                raise ValueError(f"{where}: missing {field}")
        name, trial, repeat = row["configuration"], row["trial"], row["repeat"]
        for field in ("trial", "repeat"):
            _number(row[field], field, where, integer=True)
        if name not in configs:
            raise ValueError(f"{where}: unexpected configuration")
        if trial not in range(5) or repeat not in range(17):
            raise ValueError(f"{where}: trial/repeat outside protocol")
        if (name, trial, repeat) in seen:
            raise ValueError(f"{where}: duplicate repeat")
        seen.add((name, trial, repeat))
        config = configs[name]
        for field, value in (("implementation", "c17"), ("method", config["method"]),
            ("n_points", config["n_points"]), ("n_queries", config["n_points"]), ("group", 8),
            ("max_neighbors", 0), ("status", "ok"), ("verify", "sample"), ("verify_checked", 512),
            ("verify_count_mismatches", 0), ("verify_list_errors", 0)):
            if row[field] != value or isinstance(row[field], bool):
                raise ValueError(f"{where}: {field} must equal {value!r}")
        for field in ("n_points", "n_queries", "verify_checked", "verify_count_mismatches", "verify_list_errors"):
            _number(row[field], field, where, integer=True)
        _number(row["radius"], "radius", where, positive=True)
        for field, value in row.items():
            if field.endswith("_ms"):
                if field == "first_answer_wall_ms" and value is None and repeat > 0:
                    continue
                _number(value, field, where)
        if not math.isclose(row["phase_total_ms"], row["index_ms"] + row["search_ms"], abs_tol=2.1e-6, rel_tol=1e-8):
            raise ValueError(f"{where}: phase_total_ms differs from index_ms + search_ms")
        process = (name, trial)
        if process in builds and row["index_ms"] != builds[process]:
            raise ValueError(f"{where}: index_ms changed within one index lifetime")
        builds[process] = row["index_ms"]
        if row["process_wall_ms"] != commands[process]["process_wall_ms"]:
            raise ValueError(f"{where}: process_wall_ms differs from command")
    expected = {(name, trial, repeat) for name in configs for trial in range(5) for repeat in range(17)}
    if seen != expected:
        raise ValueError(f"missing/incomplete lifecycle repeats: {len(expected-seen)} rows missing")
    return rows


def _stats(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def summarize_rows(rows, configs):
    result = []
    for config in configs:
        trials = []
        for trial in range(5):
            selected = {r["repeat"]: r for r in rows
                        if r["configuration"] == config["name"] and r["trial"] == trial}
            if set(selected) != set(range(17)):
                raise ValueError(f"incomplete lifecycle repeats for {config['name']} trial={trial}")
            trials.append(selected)
        prefixes = []
        for count in PREFIXES:
            values = {name: [] for name in ("cold_phase_ms", "warm_phase_ms", "cold_search_ms", "warm_search_ms")}
            for trial in trials:
                cold = sum(trial[i]["search_ms"] for i in range(count))
                warm = sum(trial[i]["search_ms"] for i in range(1, count + 1))
                values["cold_search_ms"].append(cold)
                values["warm_search_ms"].append(warm)
                values["cold_phase_ms"].append(trial[0]["index_ms"] + cold)
                values["warm_phase_ms"].append(trial[0]["index_ms"] + warm)
            prefixes.append(dict(batches=count, **{name: _stats(v) for name, v in values.items()}))
        result.append(dict(configuration=config["name"], method=config["method"], n=5,
                           index_ms=_stats([trial[0]["index_ms"] for trial in trials]),
                           first_answer_wall_ms=(_stats([trial[0]["first_answer_wall_ms"] for trial in trials])
                               if all(trial[0].get("first_answer_wall_ms") is not None for trial in trials) else None),
                           prefixes=prefixes))
    return result


def validate_manifest(path):
    path = Path(path)
    folder = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
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
        raise ValueError("summary.json differs from recomputed lifecycle prefixes")
    return dict(manifest=manifest, rows=rows, summary=summary)


def run_trials(executable, output, provenance):
    executable, output = Path(executable).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence directory: {output}")
    check_build_provenance(provenance, executable)
    configs = configurations()
    manifest = dict(schema_version=1, evidence_type="c17-reused-index-batches",
        created_utc=datetime.now(timezone.utc).isoformat(), random_seed=20261030,
        configurations=configs, commands=[], trials_per_configuration=5,
        repeat_ids=list(range(17)), prefix_batches=list(PREFIXES),
        binary_sha256=digest(executable), source_tree_sha256=source_tree_hash(),
        build_provenance=provenance, binary_path=str(executable),
        environment={"platform": platform.platform(), "python": platform.python_version(),
                     "gpu": subprocess.check_output(["nvidia-smi"], text=True, timeout=30)},
        timing_contract={"index_ms": "one index built per process, reused for all 17 batches",
            "cold_phase_ms(N)": "derived index_ms + sum(search_ms repeats 0..N-1)",
            "warm_phase_ms(N)": "derived index_ms + sum(search_ms repeats 1..N); repeat0 excluded",
            "first_answer_wall_ms": "repeat0 construction-start to first-query completion before verification; excludes input generation and process startup",
            "process_wall_ms": "all 17 batches plus startup, dataset creation, verification and teardown",
            "scope": "fixed identical point/query datasets; verification outside search timers; phase sums are not independently measured whole-process first-answer latency"})
    output.mkdir(parents=True)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in source_files(ROOT):
            archive.write(source, source.relative_to(ROOT).as_posix())
        for source in (Path(__file__), Path(__file__).with_name("c17_evidence.py")):
            archive.write(source, "bench/" + source.name)
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
        if len(emitted) != 17 or sorted(r.get("repeat", -1) for r in emitted) != list(range(17)):
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
    validate_rows(rows, manifest)
    partial.replace(output / "trials.jsonl")
    manifest["trials_sha256"] = digest(output / "trials.jsonl")
    (output / "summary.json").write_text(json.dumps(summarize_rows(rows, configs), indent=2, allow_nan=False), encoding="utf-8")
    candidate = output / "manifest.partial.json"
    candidate.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(candidate)
    candidate.replace(output / "manifest.json")
    print(f"Published {len(rows)} validated reused-index C17 rows in {output}", flush=True)
    return validate_manifest(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, metavar="MANIFEST")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--binary", type=Path, default=os.environ.get("OOC_RTNN", str(ROOT / "build_c17/ooc_rtnn.exe")))
    parser.add_argument("--build-record", type=Path, default=ROOT / "build_c17/build_provenance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/c17_lifecycle_20261029")
    args = parser.parse_args()
    try:
        if args.run:
            run_trials(args.binary, args.output, json.loads(args.build_record.read_text(encoding="utf-8-sig")))
        elif args.validate:
            result = validate_manifest(args.validate)
            print(f"C17 lifecycle: {len(result['rows'])} rows, 6 configurations, all prefixes/hashes/verification valid")
        else:
            parser.error("select --run or --validate MANIFEST")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Lifecycle validation failed: {error}\n")


if __name__ == "__main__":
    main()
