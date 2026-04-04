#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent

DEFAULT_TILEFLOW_CSV = TUTORIAL_DIR / "results" / "mnk_max_block_span_512_2048_2stage_compare.csv"
DEFAULT_LLMCOMPASS_CSV = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/"
    "Experiment/H100/GEMM/results/Multiple_SM_BMBNBK12825664_PERSISTENT/"
    "mnk_max_block_span_512_2048_2stage_compare.csv"
)
DEFAULT_OUT = TUTORIAL_DIR / "results" / "h100_gemm_cycle_comparison.png"
DEFAULT_MERGED = TUTORIAL_DIR / "results" / "h100_gemm_llmcompass_comparison.csv"


def load_compare_csv(csv_path, pred_key_name):
    rows = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            rows[key] = {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": int(float(row["actual_cycles"])),
                pred_key_name: int(float(row["sim_cycles"])),
            }
    return rows


def merge_rows(tileflow_rows, llmcompass_rows):
    keys = sorted(set(tileflow_rows) & set(llmcompass_rows))
    merged = []
    for key in keys:
        t = tileflow_rows[key]
        l = llmcompass_rows[key]
        actual_t = t["actual_cycles"]
        actual_l = l["actual_cycles"]
        if actual_t != actual_l:
            raise ValueError(f"Actual cycles mismatch for {key}: {actual_t} vs {actual_l}")
        merged.append(
            {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": actual_t,
                "tileflow_sim_cycles": t["tileflow_sim_cycles"],
                "llmcompass_cycles": l["llmcompass_cycles"],
            }
        )
    return merged


def ratio_stats(rows, pred_key, base_key):
    ratios = [row[pred_key] / row[base_key] for row in rows]
    return mean(ratios), median(ratios), min(ratios), max(ratios)


def write_merged_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "M",
                "N",
                "K",
                "actual_cycles",
                "tileflow_sim_cycles",
                "llmcompass_cycles",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow-csv", default=str(DEFAULT_TILEFLOW_CSV))
    parser.add_argument("--llmcompass-csv", default=str(DEFAULT_LLMCOMPASS_CSV))
    parser.add_argument("--merged-csv", default=str(DEFAULT_MERGED))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    tileflow_rows = load_compare_csv(args.tileflow_csv, "tileflow_sim_cycles")
    llmcompass_rows = load_compare_csv(args.llmcompass_csv, "llmcompass_cycles")
    rows = merge_rows(tileflow_rows, llmcompass_rows)
    write_merged_csv(rows, Path(args.merged_csv))

    rows_sorted = sorted(rows, key=lambda row: (row["actual_cycles"], row["M"], row["N"], row["K"]))

    workloads = [f"{row['M']}x{row['N']}x{row['K']}" for row in rows_sorted]
    x = np.arange(len(rows_sorted))
    actual = np.array([row["actual_cycles"] for row in rows_sorted])
    tileflow = np.array([row["tileflow_sim_cycles"] for row in rows_sorted])
    llmcompass = np.array([row["llmcompass_cycles"] for row in rows_sorted])

    tile_mean, tile_median, _, _ = ratio_stats(rows, "tileflow_sim_cycles", "actual_cycles")
    llm_mean, llm_median, _, _ = ratio_stats(rows, "llmcompass_cycles", "actual_cycles")

    by_k = {}
    for row in rows:
        by_k.setdefault(row["K"], []).append(row)

    k_values = sorted(by_k.keys())
    tile_k_means = [mean(r["tileflow_sim_cycles"] / r["actual_cycles"] for r in by_k[k]) for k in k_values]
    llm_k_means = [mean(r["llmcompass_cycles"] / r["actual_cycles"] for r in by_k[k]) for k in k_values]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    ax0.plot(x, actual, color="#1f2937", linewidth=2.5, marker="o", markersize=3, label="Actual")
    ax0.plot(x, tileflow, color="#2563eb", linewidth=2.0, marker="s", markersize=3, label="TileFlow")
    ax0.plot(x, llmcompass, color="#dc2626", linewidth=2.0, marker="^", markersize=3, label="LLMCompass")
    ax0.set_title("H100 GEMM Cycle Comparison Across 64 Workloads", fontsize=16, pad=14)
    ax0.set_ylabel("Cycles")
    ax0.set_xlabel("Workloads Sorted by Actual Cycles")
    ax0.set_xticks(x[::4])
    ax0.set_xticklabels([workloads[i] for i in range(0, len(workloads), 4)], rotation=45, ha="right", fontsize=8)
    ax0.legend(loc="upper left", ncol=3, frameon=True)
    ax0.text(
        0.99,
        0.02,
        f"TileFlow/Actual: mean {tile_mean:.3f}x, median {tile_median:.3f}x\n"
        f"LLMCompass/Actual: mean {llm_mean:.3f}x, median {llm_median:.3f}x",
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    max_cycle = max(actual.max(), tileflow.max(), llmcompass.max()) * 1.05

    ax1.scatter(actual, tileflow, color="#2563eb", alpha=0.8, s=45)
    ax1.plot([0, max_cycle], [0, max_cycle], linestyle="--", color="#6b7280", linewidth=1.5)
    ax1.set_title("TileFlow vs Actual")
    ax1.set_xlabel("Actual Cycles")
    ax1.set_ylabel("TileFlow Cycles")
    ax1.set_xlim(0, max_cycle)
    ax1.set_ylim(0, max_cycle)

    ax2.scatter(actual, llmcompass, color="#dc2626", alpha=0.8, s=45)
    ax2.plot([0, max_cycle], [0, max_cycle], linestyle="--", color="#6b7280", linewidth=1.5)
    ax2.set_title("LLMCompass vs Actual")
    ax2.set_xlabel("Actual Cycles")
    ax2.set_ylabel("LLMCompass Cycles")
    ax2.set_xlim(0, max_cycle)
    ax2.set_ylim(0, max_cycle)

    width = 0.35
    k_pos = np.arange(len(k_values))
    ax3.bar(k_pos - width / 2, tile_k_means, width, color="#2563eb", label="TileFlow")
    ax3.bar(k_pos + width / 2, llm_k_means, width, color="#dc2626", label="LLMCompass")
    ax3.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.2)
    ax3.set_title("Mean Predicted / Actual by K")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Mean Ratio")
    ax3.set_xticks(k_pos)
    ax3.set_xticklabels([str(k) for k in k_values])
    ax3.legend(frameon=True)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    print(f"Saved merged CSV: {args.merged_csv}")
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
