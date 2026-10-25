"""Turns results/raw/*.jsonl into figures (results/figures/*.png) and tables (results/summary.md).

    python bench/plot_results.py

Timings of repeated runs are reduced to the best (minimum) run.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import FIGURES, RAW, RESULTS, load_frame  # noqa: E402

# ---- style (validated categorical palette; colour follows the method everywhere) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_STYLE = {  # label, colour, marker
    "rt-ooc": ("RT out-of-core (ours)", SERIES[0], "o"),
    "rt-incore/8": ("RT in-core, G=8", SERIES[1], "s"),
    "rt-incore/1": ("RT in-core, G=1 (RTNN-style)", SERIES[2], "^"),
    "cuda-grid": ("CUDA uniform grid", SERIES[3], "D"),
    "cpu-grid": ("CPU grid (OpenMP)", SERIES[4], "v"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.axisbelow": True, "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8.5, "legend.labelcolor": INK2, "lines.linewidth": 1.6, "lines.markersize": 5.5,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "figure.dpi": 110,
})


def key_method(row) -> str:
    m = row["method"]
    return f"{m}/{int(row.get('group', 1))}" if m == "rt-incore" else m


def best(df: pd.DataFrame, by: list[str], col: str = "search_ms") -> pd.DataFrame:
    ok = df[df["status"] == "ok"] if "status" in df else df
    return ok.groupby(by, dropna=False, as_index=False)[col].min()


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def dataset_name(spec: str) -> str:
    return spec.split(":")[0]


tables: list[str] = []


def table(title: str, df: pd.DataFrame, floatfmt: str = ".2f"):
    tables.append(f"### {title}\n\n{df.to_markdown(index=False, floatfmt=floatfmt)}\n")


# ---------------------------------------------------------------------------
def fig_baselines():
    df = load_frame(RAW / "baselines.jsonl")
    if df.empty:
        return
    df["key"] = df.apply(key_method, axis=1)
    df["dataset"] = df["points_spec"].map(dataset_name)
    b = best(df, ["dataset", "key"])
    b["mq_s"] = df["n_queries"].iloc[0] / (b["search_ms"] * 1e3)
    order = [k for k in METHOD_STYLE if k in set(b["key"])]
    sets = list(dict.fromkeys(df["dataset"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(4.1 * len(sets), 3.0), sharey=True)
    for ax, ds in zip(axes, sets):
        sub = b[b["dataset"] == ds].set_index("key").reindex(order)
        y = range(len(order))
        ax.barh(y, sub["mq_s"], color=[METHOD_STYLE[k][1] for k in order], height=0.62, edgecolor=SURFACE,
                linewidth=1.5)
        for yi, v in zip(y, sub["mq_s"]):
            if not math.isnan(v):
                ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(list(y), [METHOD_STYLE[k][0] for k in order])
        ax.invert_yaxis()
        ax.set_title(f"{ds} (4M points, Q = P)")
        ax.set_xlabel("throughput, million queries / s (higher is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, sub["mq_s"].max() * 1.25)
    save(fig, "baselines_throughput")
    piv = b.pivot(index="key", columns="dataset", values="search_ms").reindex(order)
    piv.index = [METHOD_STYLE[k][0] for k in piv.index]
    table("Baselines: end-to-end search time (ms, best of 3)", piv.reset_index(names="method"))
    ok = df[df["status"] == "ok"].copy()
    ok["device_ms"] = ok["query_ms"].fillna(ok["traverse_ms"])  # out-of-core: traversal summed over tiles
    q = ok.groupby(["dataset", "key"], as_index=False)[["device_ms"]].min()
    if not q.empty:
        qp = q.pivot(index="key", columns="dataset", values="device_ms").reindex(order)
        qp.index = [METHOD_STYLE[k][0] for k in qp.index]
        table("Baselines: device query kernel / traversal time only (ms)", qp.reset_index(names="method"))


def fig_scaling():
    df = load_frame(RAW / "scaling.jsonl")
    if df.empty:
        return
    df["key"] = df.apply(key_method, axis=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rows = []
    top = df.loc[df["status"] == "ok", "search_ms"].max()
    for key, (label, color, marker) in METHOD_STYLE.items():
        sub = df[df["key"] == key]
        if sub.empty:
            continue
        ok = sub[sub["status"] == "ok"].groupby("n_points", as_index=False)["search_ms"].min()
        ax.plot(ok["n_points"] / 1e6, ok["search_ms"] / 1e3, marker=marker, color=color, label=label)
        failed = sorted(set(sub.loc[sub["status"] != "ok", "n_points"]) - set(ok["n_points"]))
        if failed:
            ax.scatter([f / 1e6 for f in failed], [top / 1e3 * 1.6] * len(failed), marker="x", color=color, s=40)
        for _, r in ok.iterrows():
            rows.append({"method": label, "N (M)": r["n_points"] / 1e6, "search (s)": r["search_ms"] / 1e3})
        for f in failed:
            rows.append({"method": label, "N (M)": f / 1e6, "search (s)": float("nan")})
    if any(r["search (s)"] != r["search (s)"] for r in rows):  # NaN marks an out-of-memory run
        ax.text(0.01, 0.98, "x = does not fit in device memory (OOM)", transform=ax.transAxes, va="top",
                fontsize=8, color=MUTED)
    # In-core working sets pass 8 GB here: WDDM then pages device memory to host
    # RAM instead of failing, and the in-core methods slow down sharply.
    ax.set_xscale("log")
    ax.set_yscale("log")
    xmin, xmax = df["n_points"].min() / 1e6 / 1.15, df["n_points"].max() / 1e6 * 1.15
    ax.set_xlim(xmin, xmax)
    ax.axvspan(80, xmax, color=GRID, alpha=0.45, linewidth=0, zorder=0)
    ax.text(84, 0.03, "in-core working set > 8 GB:\ndriver pages to host RAM", fontsize=8, color=INK2,
            va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("points N = queries M (millions, uniform, ~32 neighbors)")
    ax.set_ylabel("end-to-end search time (s)")
    ax.set_title("Scaling past device memory (RTX 4060 Laptop, 8 GB)")
    ax.legend(loc="upper left")
    save(fig, "scaling")
    t = pd.DataFrame(rows).pivot_table(index="N (M)", columns="method", values="search (s)", dropna=False)
    table("Scaling: end-to-end search time in seconds (no run ran out of memory: past 8 GB the driver pages)",
          t.reset_index())


def fig_tile_size():
    df = load_frame(RAW / "tile_size.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["group", "tile_depth", "tiles"], as_index=False).agg(
        exec_ms=("exec_ms", "min"), predicted_ms=("predicted_ms", "first"), halo=("halo_fraction", "first"),
        tile_mb=("tile_store_bytes", "first"))
    groups = sorted(g["group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.3), sharey=True)
    axes = axes if len(groups) > 1 else [axes]
    for ax, grp in zip(axes, groups):
        sub = g[g["group"] == grp]
        ax.plot(sub["tiles"], sub["exec_ms"], marker="o", color=SERIES[0], label="measured")
        ax.plot(sub["tiles"], sub["predicted_ms"], marker="s", color=SERIES[1], label="cost model")
        best_row = sub.loc[sub["exec_ms"].idxmin()]
        ax.annotate("fastest", (best_row["tiles"], best_row["exec_ms"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tiles (kd depth 2..10)")
        ax.set_title(f"G = {int(grp)} points per primitive")
    axes[0].set_ylabel("pipelined execution time (ms)")
    axes[0].legend()
    save(fig, "tile_size_model")
    g["error %"] = 100 * (g["predicted_ms"] - g["exec_ms"]) / g["exec_ms"]
    table("Tile size: measured vs predicted execution time (20M points, 512 MB budget)",
          g[["group", "tile_depth", "tiles", "halo", "exec_ms", "predicted_ms", "error %"]])


def fig_order():
    df = load_frame(RAW / "ablation_order.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["order", "eviction"], as_index=False).agg(
        loads=("loads_per_tile", "first"), h2d_mb=("h2d_bytes", "first"), exec_ms=("exec_ms", "min"),
        search_ms=("search_ms", "min"))
    g["h2d_mb"] /= 2 ** 20
    orders = [o for o in ("tiled", "morton", "random", "original") if o in set(g["order"])]
    evs = ["belady", "lru"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    width = 0.38
    for i, ev in enumerate(evs):
        sub = g[g["eviction"] == ev].set_index("order").reindex(orders)
        xs = [x + (i - 0.5) * width for x in range(len(orders))]
        for ax, col in zip(axes, ("loads", "exec_ms")):
            ax.bar(xs, sub[col], width=width, color=SERIES[i], label="Belady (offline optimal)" if ev == "belady"
                   else "LRU", edgecolor=SURFACE, linewidth=1.5)
            for x, v in zip(xs, sub[col]):
                ax.text(x, v, f"{v:.1f}" if col == "loads" else f"{v:.0f}", ha="center", va="bottom", fontsize=7.5,
                        color=INK2)
    for ax, title, ylabel in ((axes[0], "Tile loads per tile (1.0 = each tile streamed once)", "loads / tile"),
                              (axes[1], "Pipelined execution time", "ms")):
        ax.set_xticks(range(len(orders)), orders)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    save(fig, "ablation_order")
    table("Query order x eviction (20M points, 128 tiles, 4 slots, 1M-query windows)",
          g[["order", "eviction", "loads", "h2d_mb", "exec_ms", "search_ms"]])


def fig_overlap():
    df = load_frame(RAW / "ablation_overlap.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["group", "slots", "overlap"], as_index=False).agg(
        exec_ms=("exec_ms", "min"), hidden=("transfer_hidden_fraction", "max"), traverse=("traverse_ms", "min"),
        h2d=("tile_h2d_ms", "min"))
    groups = sorted(g["group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.2), sharey=False)
    axes = axes if len(groups) > 1 else [axes]
    for ax, grp in zip(axes, groups):
        for i, (ov, label) in enumerate(((1.0, "copy/compute overlap"), (0.0, "serialized (one stream)"))):
            sub = g[(g["group"] == grp) & (g["overlap"] == ov)]
            ax.plot(sub["slots"], sub["exec_ms"], marker="o", color=SERIES[i], label=label)
        trav = g[(g["group"] == grp)]["traverse"].min()
        ax.axhline(trav, color=MUTED, linewidth=1)
        ax.text(g["slots"].max(), trav, "traversal alone", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_xlabel("tile slots on the device")
        ax.set_title(f"G = {int(grp)}")
    axes[0].set_ylabel("pipelined execution time (ms)")
    axes[0].legend()
    save(fig, "ablation_overlap")
    table("Double buffering: slots x overlap (20M points, 128 tiles)", g)


def fig_budget():
    df = load_frame(RAW / "budget.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    ooc = ok[ok["method"] == "rt-ooc"].copy()
    ooc["budget_mb"] = ooc["device_budget_bytes"] / 2 ** 20
    ooc = ooc.groupby("budget_mb", as_index=False).agg(
        search_ms=("search_ms", "min"), exec_ms=("exec_ms", "min"), tiles=("tiles", "first"),
        slots=("slots", "first"), window_queries=("window_queries", "first"), predicted_ms=("predicted_ms", "first"))
    inc = ok[ok["method"] == "rt-incore"]["search_ms"].min()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ooc["budget_mb"], ooc["search_ms"], marker="o", color=SERIES[0], label=METHOD_STYLE["rt-ooc"][0])
    ax.plot(ooc["budget_mb"], ooc["exec_ms"], marker="s", color=SERIES[6], label="its GPU pipeline alone")
    if not math.isnan(inc):
        ax.axhline(inc, color=SERIES[1], linewidth=1.4, label=METHOD_STYLE["rt-incore/8"][0] + " (all memory)")
    for _, r in ooc.iterrows():
        ax.annotate(f"{int(r['tiles'])} tiles", (r["budget_mb"], r["search_ms"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, None)
    ax.set_xlabel("device memory budget (MB)")
    ax.set_ylabel("time (ms)")
    ax.set_title("Emulating smaller GPUs: 40M points, automatic tiling")
    ax.legend(loc="lower right")
    save(fig, "budget")
    table("Device budget sweep (40M points)",
          ooc[["budget_mb", "tiles", "slots", "window_queries", "search_ms", "exec_ms", "predicted_ms"]])


def fig_group():
    df = load_frame(RAW / "group_size.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"].copy()
    ok["dataset"] = ok["points_spec"].map(dataset_name)
    inc = ok[ok["method"] == "rt-incore"].groupby(["dataset", "group"], as_index=False).agg(
        query_ms=("query_ms", "min"), gas=("gas_bytes_per_point", "first"))
    ooc = ok[ok["method"] == "rt-ooc"].groupby(["dataset", "group"], as_index=False).agg(
        exec_ms=("exec_ms", "min"), store=("tile_store_bytes", "first"), n=("n_points", "first"))
    sets = list(dict.fromkeys(ok["dataset"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for i, ds in enumerate(sets):
        sub = inc[inc["dataset"] == ds]
        base = sub.loc[sub["group"] == 1, "query_ms"]
        axes[0].plot(sub["group"], sub["query_ms"] / (base.iloc[0] if len(base) else 1), marker="o", color=SERIES[i],
                     label=ds)
        so = ooc[ooc["dataset"] == ds]
        axes[1].plot(so["group"], so["store"] / so["n"], marker="o", color=SERIES[i], label=ds)
    axes[0].axhline(1.0, color=MUTED, linewidth=1)
    axes[0].set_title("In-core traversal time vs G = 1")
    axes[0].set_ylabel("relative time (lower is better)")
    axes[1].set_title("Tile store bytes per point (what is streamed)")
    axes[1].set_ylabel("bytes / point")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("points per BVH primitive (G)")
    axes[0].legend()
    save(fig, "group_size")
    m = inc.merge(ooc, on=["dataset", "group"], how="outer")
    m["store B/pt"] = m["store"] / m["n"]
    table("Point bucketing (4M points)", m[["dataset", "group", "query_ms", "gas", "exec_ms", "store B/pt"]])


def fig_halo():
    df = load_frame(RAW / "halo.jsonl")
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["halo", "tile_depth", "radius"], as_index=False).agg(
        k=("neighbors_avg", "first"), exec_ms=("exec_ms", "min"), dup=("predicted_dup_factor", "first"),
        h2d=("tile_load_bytes", "first"), pairs=("pairs_per_query", "first"))
    depths = sorted(g["tile_depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(4.4 * len(depths), 3.2), sharey=True)
    axes = axes if len(depths) > 1 else [axes]
    for ax, d in zip(axes, depths):
        for i, mode in enumerate(("points", "queries")):
            sub = g[(g["tile_depth"] == d) & (g["halo"] == mode)].sort_values("k")
            ax.plot(sub["k"], sub["exec_ms"], marker="o", color=SERIES[i],
                    label="halo points (duplicate data)" if mode == "points" else "halo queries (duplicate rays)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("mean neighbors per query")
        ax.set_title(f"{2 ** int(d)} tiles")
    axes[0].set_ylabel("pipelined execution time (ms)")
    axes[0].legend()
    save(fig, "halo_modes")
    table("Halo strategy (10M points)", g.sort_values(["tile_depth", "k", "halo"]))


def fig_lists():
    df = load_frame(RAW / "lists.jsonl")
    if df.empty:
        return
    df["key"] = df.apply(key_method, axis=1)
    b = best(df, ["max_neighbors", "key"])
    b["mq_s"] = df["n_queries"].iloc[0] / (b["search_ms"] * 1e3)
    table("Neighbor lists: search time (ms) and throughput", b)


def main():
    for f in (fig_baselines, fig_scaling, fig_tile_size, fig_order, fig_overlap, fig_budget, fig_group, fig_halo,
              fig_lists):
        try:
            f()
        except Exception as e:  # keep going: one broken experiment must not hide the others
            print(f"skipped {f.__name__}: {e!r}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("# Measured results\n\nGenerated by `bench/plot_results.py` from "
                                        "`results/raw/*.jsonl`. Times are the best of the repeated runs.\n\n"
                                        + "\n".join(tables), encoding="utf-8")
    print("wrote", RESULTS / "summary.md")


if __name__ == "__main__":
    main()
