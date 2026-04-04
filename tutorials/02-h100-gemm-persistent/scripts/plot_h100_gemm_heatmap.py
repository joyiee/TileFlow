#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent

DEFAULT_MERGED_CSV = TUTORIAL_DIR / "results" / "h100_gemm_llmcompass_comparison.csv"
DEFAULT_OUT = TUTORIAL_DIR / "results" / "h100_gemm_ratio_heatmap.png"


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k: int(v) for k, v in row.items()}
            row["tileflow_ratio"] = row["tileflow_sim_cycles"] / row["actual_cycles"]
            row["llmcompass_ratio"] = row["llmcompass_cycles"] / row["actual_cycles"]
            rows.append(row)
    return rows


def build_matrix(rows, k_value, ratio_key, m_values, n_values):
    mat = np.full((len(m_values), len(n_values)), np.nan)
    row_map = {(r["M"], r["N"], r["K"]): r for r in rows}
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            mat[i, j] = row_map[(m, n, k_value)][ratio_key]
    return mat


def annotate(ax, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 1.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_MERGED_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = load_rows(args.csv)
    m_values = sorted({r["M"] for r in rows})
    n_values = sorted({r["N"] for r in rows})
    k_values = sorted({r["K"] for r in rows})

    fig, axes = plt.subplots(2, len(k_values), figsize=(4.2 * len(k_values), 8.2), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=2.4)

    row_specs = [
        ("tileflow_ratio", "TileFlow / Actual"),
        ("llmcompass_ratio", "LLMCompass / Actual"),
    ]

    last_im = None
    for row_idx, (ratio_key, row_title) in enumerate(row_specs):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            mat = build_matrix(rows, k, ratio_key, m_values, n_values)
            last_im = ax.imshow(mat, cmap=cmap, norm=norm, origin="upper")
            annotate(ax, mat)
            ax.set_title(f"{row_title}, K={k}", fontsize=11)
            ax.set_xticks(range(len(n_values)))
            ax.set_xticklabels(n_values)
            ax.set_yticks(range(len(m_values)))
            ax.set_yticklabels(m_values)
            ax.set_xlabel("N")
            if col_idx == 0:
                ax.set_ylabel("M")

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("Predicted / Actual")

    fig.suptitle("H100 GEMM Ratio Heatmap by M/N/K", fontsize=16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
