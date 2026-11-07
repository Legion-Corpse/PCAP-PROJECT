"""Write results/summary.md from the validated measurement manifests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from c17_evidence import validate_manifest
from c17_lifecycle import validate_manifest as validate_lifecycle


def main():
    release = validate_manifest(ROOT / "results/c17_20261029_v2/manifest.json")
    lookup = {row["configuration"]: row for row in release["summary"]}
    lines = [
        "# Measured results",
        "",
        "Generated from the validated primary and lifecycle manifests. Time values are medians in milliseconds unless noted. The main matrix uses five independent processes per configuration. Each row has sampled verification for 512 deterministic query positions. Cold first answer ends before verification; retained search reuses the index and includes that backend's query ordering, routing, transfer, execution, and result handling.",
        "",
        "The test machine was a Windows 11 laptop with an RTX 4060 Laptop GPU and 8 GB device memory. GPU clocks were not locked. The serial CPU grid is a reference baseline, not a tuned multicore implementation.",
        "",
        "## Primary comparison",
        "",
        "| Workload | Backend | Build | Retained search | Cold first answer |",
        "|---|---|---:|---:|---:|",
    ]
    workloads = [
        ("uniform100K", "Uniform 100K"),
        ("uniform1M", "Uniform 1M"),
        ("uniform4M", "Uniform 4M"),
        ("scene200K", "Scene-like 200K"),
        ("armadillo", "Armadillo 172,974"),
    ]
    for key, label in workloads:
        for method in ("cpu-grid", "cuda-grid", "rt-incore", "rt-ooc"):
            row = lookup[f"{key}_{method}"]
            lines.append(
                f"| {label} | {method} | {row['index_ms']['median']:.2f} | "
                f"{row['search_ms']['median']:.2f} | {row['first_answer_wall_ms']['median']:.2f} |"
            )
    lines += [
        "",
        "The CUDA linked-cell grid has the shortest retained-search median in all five tested workloads. This is a result for the implementations and machine above; it does not compare against an optimized multicore CPU library or other GPUs.",
        "",
        "## Streamed RT capacity and planner",
        "",
        "| 1M configuration | Retained search | Observed range |",
        "|---|---:|---:|",
    ]
    for row in release["summary"]:
        if row["configuration"].startswith("uniform1M_rt-ooc"):
            metric = row["search_ms"]
            name = row["configuration"].replace("uniform1M_rt-ooc_", "")
            if name == row["configuration"]:
                name = "64K/on (default)"
            name = name.replace("tile", "").replace("_overlap", "/")
            name = name.replace("auto", "automatic")
            lines.append(f"| {name} | {metric['median']:.2f} | {metric['min']:.2f} to {metric['max']:.2f} |")
    lines += [
        "",
        "Automatic selection uses a finite calibration set for same-point, count-only queries. Calibration time is part of setup. Predictions exclude host sorting, routing, and scatter. The 1M automatic trial selected a one-million-point tile, so this result is not evidence of a multi-tile streaming speedup.",
        "",
        "## Explicit allocation quantities at 4M points",
        "",
        "| Backend | Executor buffers MiB | Pinned tile store MiB |",
        "|---|---:|---:|",
    ]
    for method in ("rt-incore", "rt-ooc"):
        row = lookup[f"uniform4M_{method}"]
        lines.append(
            f"| {method} | {row['executor_bytes']['median']/1048576:.3f} | "
            f"{row['host_tile_bytes']['median']/1048576:.3f} |"
        )
    lines += [
        "",
        "Executor bytes do not include context state, construction, calibration, input arrays, sorting temporaries, or total process memory. The Armadillo reconstruction fits physical device memory and does not establish real-data performance beyond VRAM.",
        "",
        "## Retained-index lifecycle",
        "",
        "Each lifecycle process builds once and makes seventeen query calls. Prefix time is summed within each trial before taking a cross-trial median. The cold prefix includes the first query call; the warm prefix excludes it. Values below include construction once and are seconds.",
        "",
        "| 4M backend | One cold batch | Sixteen cold batches | One warm batch | Sixteen warm batches |",
        "|---|---:|---:|---:|---:|",
    ]
    life = validate_lifecycle(ROOT / "results/c17_lifecycle_20261029/manifest.json")
    for row in life["summary"]:
        if "4M" not in row["configuration"]:
            continue
        cold = row["prefixes"]
        warm = row["prefixes"]
        lines.append(
            f"| {row['method']} | {cold[0]['cold_phase_ms']['median']/1000:.3f} | "
            f"{cold[-1]['cold_phase_ms']['median']/1000:.3f} | "
            f"{warm[0]['warm_phase_ms']['median']/1000:.3f} | "
            f"{warm[-1]['warm_phase_ms']['median']/1000:.3f} |"
        )
    lines += [
        "",
        "Lifecycle results describe repeated queries against a fixed retained index. They do not claim a universal amortization break-even point.",
        "",
        "## Reproducibility and detailed results",
        "",
        "The primary release contains 250 rows from 125 independent processes and 25 configurations. The lifecycle release contains 510 rows from 30 processes and six configurations. Validators recompute summaries and verify source, executable, and raw-result hashes.",
        "",
        "See the [primary manifest](c17_20261029_v2/manifest.json) and [lifecycle manifest](c17_lifecycle_20261029/manifest.json) for the measured build identities, source snapshots, and raw data.",
        "",
    ]
    target = ROOT / "results/summary.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated measured-results summary: {target}")


if __name__ == "__main__":
    main()
