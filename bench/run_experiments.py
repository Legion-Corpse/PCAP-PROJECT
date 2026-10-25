"""Runs the experiment suite and writes JSON lines to results/raw/<experiment>.jsonl.

    python bench/run_experiments.py              # everything (~30-40 min on an RTX 4060 laptop)
    python bench/run_experiments.py --quick      # small sizes, a few minutes
    python bench/run_experiments.py scaling ablation_order

Each experiment answers one question from the synopsis; see docs/EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import time

from common import RAW, run

QUICK = False


def size(full: str, quick: str) -> str:
    return quick if QUICK else full


def out(name: str):
    path = RAW / f"{name}.jsonl"
    if path.exists():
        path.unlink()
    return path


# --------------------------------------------------------------------------
# E1: every backend on three data distributions (same answers, verified).
# --------------------------------------------------------------------------
def baselines():
    p = out("baselines")
    n = size("4M", "500K")
    for spec, radius in ((f"uniform:{n}", None), (f"clustered:{n}", "0.004"), (f"scene:{n}", "0.00276")):
        r = ["--radius", radius] if radius else ["--k-expected", "32"]
        common = ["--points", spec, *r, "--repeat", "3", "--verify", "sample"]
        run([*common, "--method", "cpu-grid,cuda-grid"], p)
        for g in ("1", "8"):
            run([*common, "--method", "rt-incore", "--group", g], p)
        run([*common, "--method", "rt-ooc", "--group", "8", "--budget-mb", "256"], p)


# --------------------------------------------------------------------------
# E2: scaling past device memory. In-core methods eventually fail (OOM);
# the out-of-core search keeps going.
# --------------------------------------------------------------------------
def scaling():
    p = out("scaling")
    sizes = ["5M", "10M", "20M", "40M", "80M", "120M", "160M"] if not QUICK else ["1M", "2M", "4M"]
    for n in sizes:
        common = ["--points", f"uniform:{n}", "--k-expected", "32", "--verify", "sample", "--verify-samples", "5000"]
        run([*common, "--method", "cuda-grid"], p)
        run([*common, "--method", "rt-incore", "--group", "1"], p)
        run([*common, "--method", "rt-incore", "--group", "8"], p)
        run([*common, "--method", "rt-ooc", "--group", "8", "--budget-mb", "1024"], p)


# --------------------------------------------------------------------------
# E3: tile size (kd depth) sweep, measured vs cost-model prediction.
# --------------------------------------------------------------------------
def tile_size():
    p = out("tile_size")
    n = size("20M", "2M")
    for g in ("1", "8"):
        for depth in range(2, 11 if not QUICK else 7):
            run(["--points", f"uniform:{n}", "--k-expected", "32", "--method", "rt-ooc", "--group", g, "--depth",
                 str(depth), "--leaf-points", "4K", "--budget-mb", "512", "--verify", "none", "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E4: query order x eviction policy with a small tile cache.
# --------------------------------------------------------------------------
def ablation_order():
    p = out("ablation_order")
    n = size("20M", "2M")
    run(["--points", f"uniform:{n}", "--k-expected", "32", "--method", "rt-ooc", "--group", "8", "--depth", "7",
         "--budget-mb", "512", "--slots", "4", "--window", size("1M", "100K"),
         "--order", "tiled,morton,random,original", "--eviction", "belady,lru", "--verify", "sample",
         "--verify-samples", "5000", "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E5: double buffering - slots and copy/compute overlap.
# --------------------------------------------------------------------------
def ablation_overlap():
    p = out("ablation_overlap")
    n = size("20M", "2M")
    for g in ("1", "8"):
        run(["--points", f"uniform:{n}", "--k-expected", "32", "--method", "rt-ooc", "--group", g, "--depth", "7",
             "--budget-mb", "1024", "--slots", "1,2,3,4,8", "--overlap", "on,off", "--verify", "none",
             "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E6: device budget sweep (emulates smaller GPUs) with automatic tiling.
# --------------------------------------------------------------------------
def budget():
    p = out("budget")
    n = size("40M", "4M")
    common = ["--points", f"uniform:{n}", "--k-expected", "32", "--verify", "sample", "--verify-samples", "5000"]
    run([*common, "--method", "rt-incore", "--group", "8", "--repeat", "2"], p)
    for mb in ("64", "128", "256", "512", "1024", "2048", "0"):
        run([*common, "--method", "rt-ooc", "--group", "8", "--budget-mb", mb, "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E7: point bucketing (points per primitive) per distribution.
# --------------------------------------------------------------------------
def group_size():
    p = out("group_size")
    n = size("4M", "500K")
    for spec, radius in ((f"uniform:{n}", None), (f"clustered:{n}", "0.004"), (f"scene:{n}", "0.00276")):
        r = ["--radius", radius] if radius else ["--k-expected", "32"]
        for g in ("1", "2", "4", "8", "16", "32"):
            run(["--points", spec, *r, "--method", "rt-incore", "--group", g, "--verify", "none", "--repeat", "3"], p)
            run(["--points", spec, *r, "--method", "rt-ooc", "--group", g, "--depth", "6", "--budget-mb", "256",
                 "--verify", "none", "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E8: halo strategy (duplicate points vs duplicate rays) as the radius grows.
# --------------------------------------------------------------------------
def halo():
    p = out("halo")
    n = size("10M", "1M")
    for k in ("8", "32", "128", "512"):
        for mode in ("points", "queries"):
            for depth in ("5", "8"):
                run(["--points", f"uniform:{n}", "--k-expected", k, "--method", "rt-ooc", "--group", "8",
                     "--depth", depth, "--halo", mode, "--budget-mb", "512", "--verify", "sample",
                     "--verify-samples", "3000", "--repeat", "2"], p)


# --------------------------------------------------------------------------
# E9: neighbor lists (K ids per query) instead of counts.
# --------------------------------------------------------------------------
def lists():
    p = out("lists")
    n = size("10M", "1M")
    for K in ("16", "64"):
        common = ["--points", f"uniform:{n}", "--k-expected", "32", "--max-neighbors", K, "--verify", "sample",
                  "--verify-samples", "5000"]
        run([*common, "--method", "cuda-grid"], p)
        run([*common, "--method", "rt-incore", "--group", "8"], p)
        run([*common, "--method", "rt-ooc", "--group", "8", "--budget-mb", "512"], p)


EXPERIMENTS = {f.__name__: f for f in (baselines, scaling, tile_size, ablation_order, ablation_overlap, budget,
                                       group_size, halo, lists)}


def main():
    global QUICK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"subset of: {', '.join(EXPERIMENTS)}")
    ap.add_argument("--quick", action="store_true", help="small problem sizes")
    a = ap.parse_args()
    QUICK = a.quick
    for name in a.names or EXPERIMENTS:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")
        t = time.time()
        print(f"=== {name} ===", flush=True)
        EXPERIMENTS[name]()
        print(f"=== {name}: {time.time() - t:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
