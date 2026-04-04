#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent

DEFAULT_TILEFLOW_CSV = TUTORIAL_DIR / "results" / "mnk_max_block_span_512_2048_2stage_compare.csv"
DEFAULT_TILEDATAFLOW_LLMCOMPASS_CSV = Path("/Users/jwhuang/Code/LLMCompass/ref/h100_gemm_llmcompass_comparison.csv")
DEFAULT_MERGED_CSV = TUTORIAL_DIR / "results" / "h100_gemm_all_models_comparison.csv"
DEFAULT_OUT = TUTORIAL_DIR / "results" / "h100_gemm_all_models_comparison.png"


def load_tileflow_rows(csv_path):
    rows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            rows[key] = {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": int(float(row["actual_cycles"])),
                "tileflow_cycles": int(float(row["sim_cycles"])),
            }
    return rows


def load_tiledataflow_llmcompass_rows(csv_path):
    rows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            rows[key] = {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": int(float(row["actual_cycles"])),
                "tiledataflow_cycles": int(float(row["tileflow_sim_cycles"])),
                "llmcompass_cycles": int(float(row["llmcompass_cycles"])),
            }
    return rows


def merge_rows(tileflow_rows, other_rows):
    keys = sorted(set(tileflow_rows) & set(other_rows))
    merged = []
    for key in keys:
        t = tileflow_rows[key]
        o = other_rows[key]
        if t["actual_cycles"] != o["actual_cycles"]:
            raise ValueError(f"Actual mismatch at {key}: {t['actual_cycles']} vs {o['actual_cycles']}")
        merged.append(
            {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": t["actual_cycles"],
                "tiledataflow_cycles": o["tiledataflow_cycles"],
                "tileflow_cycles": t["tileflow_cycles"],
                "llmcompass_cycles": o["llmcompass_cycles"],
            }
        )
    return merged


def write_merged_csv(rows, out_path):
    fieldnames = [
        "M",
        "N",
        "K",
        "actual_cycles",
        "tiledataflow_cycles",
        "tileflow_cycles",
        "llmcompass_cycles",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def add_ratio(rows):
    for row in rows:
        actual = row["actual_cycles"]
        row["tiledataflow_ratio"] = row["tiledataflow_cycles"] / actual
        row["tileflow_ratio"] = row["tileflow_cycles"] / actual
        row["llmcompass_ratio"] = row["llmcompass_cycles"] / actual
    return rows


def ratio_text(rows, ratio_key):
    vals = [row[ratio_key] for row in rows]
    return f"{mean(vals):.3f}x"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow-csv", default=str(DEFAULT_TILEFLOW_CSV))
    parser.add_argument("--tiledataflow-llmcompass-csv", default=str(DEFAULT_TILEDATAFLOW_LLMCOMPASS_CSV))
    parser.add_argument("--merged-csv", default=str(DEFAULT_MERGED_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    tileflow_rows = load_tileflow_rows(args.tileflow_csv)
    other_rows = load_tiledataflow_llmcompass_rows(args.tiledataflow_llmcompass_csv)
    rows = add_ratio(merge_rows(tileflow_rows, other_rows))
    write_merged_csv(rows, Path(args.merged_csv))

    rows_sorted = sorted(rows, key=lambda row: (row["actual_cycles"], row["M"], row["N"], row["K"]))
    x = np.arange(len(rows_sorted))
    workloads = [f"{r['M']}x{r['N']}x{r['K']}" for r in rows_sorted]
    actual = np.array([r["actual_cycles"] for r in rows_sorted])
    tiledataflow = np.array([r["tiledataflow_cycles"] for r in rows_sorted])
    tileflow = np.array([r["tileflow_cycles"] for r in rows_sorted])
    llmcompass = np.array([r["llmcompass_cycles"] for r in rows_sorted])

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        gridspec_kw={"height_ratios": [1.5, 1]},
        constrained_layout=True,
    )

    ax0.plot(x, actual, color="#111827", linewidth=2.2, marker="o", markersize=3.5, label="Actual")
    ax0.plot(x, tiledataflow, color="#1d4ed8", linewidth=1.8, marker="s", markersize=3.2, label="TileDataflow")
    ax0.plot(x, tileflow, color="#f59e0b", linewidth=1.8, marker="D", markersize=3.2, label="TileFlow")
    ax0.plot(x, llmcompass, color="#dc2626", linewidth=1.8, marker="^", markersize=3.2, label="LLMCompass")
    ax0.set_title("H100 GEMM: Actual vs TileDataflow vs TileFlow vs LLMCompass", fontsize=16, pad=12)
    ax0.set_ylabel("Cycles")
    ax0.set_xlabel("Workloads Sorted by Actual Cycles")
    ax0.set_xticks(x[::4])
    ax0.set_xticklabels([workloads[i] for i in range(0, len(workloads), 4)], rotation=45, ha="right", fontsize=8)
    ax0.legend(loc="upper left", ncol=4, frameon=True)
    ax0.grid(alpha=0.25)

    ax1.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.2)
    ax1.plot(x, [r["tiledataflow_ratio"] for r in rows_sorted], color="#1d4ed8", marker="s", markersize=3.2, linewidth=1.5, label=f"TileDataflow / Actual (mean {ratio_text(rows_sorted, 'tiledataflow_ratio')})")
    ax1.plot(x, [r["tileflow_ratio"] for r in rows_sorted], color="#f59e0b", marker="D", markersize=3.2, linewidth=1.5, label=f"TileFlow / Actual (mean {ratio_text(rows_sorted, 'tileflow_ratio')})")
    ax1.plot(x, [r["llmcompass_ratio"] for r in rows_sorted], color="#dc2626", marker="^", markersize=3.2, linewidth=1.5, label=f"LLMCompass / Actual (mean {ratio_text(rows_sorted, 'llmcompass_ratio')})")
    ax1.set_ylabel("Predicted / Actual")
    ax1.set_xlabel("Workloads Sorted by Actual Cycles")
    ax1.set_xticks(x[::4])
    ax1.set_xticklabels([workloads[i] for i in range(0, len(workloads), 4)], rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(alpha=0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    print(f"Saved merged CSV: {args.merged_csv}")
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
