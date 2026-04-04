#!/usr/bin/env python3
import csv
import math
import os
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
RESULT_DIR = TUTORIAL_DIR / "results"
os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(RESULT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIOS = [
    {
        "id": "fa2",
        "title": "FA2",
        "base_csv": RESULT_DIR / "attention_family_all_models.csv",
    },
    {
        "id": "flashdecoding",
        "title": "FlashDecoding",
        "base_csv": RESULT_DIR / "attention_family_all_models.csv",
    },
    {
        "id": "flashmla",
        "title": "FlashMLA",
        "base_csv": RESULT_DIR / "attention_family_all_models.csv",
    },
]

MODEL_COLORS = {
    "TileDataflow": "#2563eb",
    "LLMCompass": "#dc2626",
    "TileFlow": "#059669",
}


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mape(actual, pred):
    return sum(abs(p - a) / a for a, p in zip(actual, pred) if a) * 100.0 / len(actual)


def corr(actual, pred):
    mean_a = sum(actual) / len(actual)
    mean_p = sum(pred) / len(pred)
    cov = sum((a - mean_a) * (p - mean_p) for a, p in zip(actual, pred))
    var_a = sum((a - mean_a) ** 2 for a in actual)
    var_p = sum((p - mean_p) ** 2 for p in pred)
    den = math.sqrt(var_a * var_p)
    return cov / den if den else 0.0


def choose_best_tileflow(rows):
    actual = [float(r["actual_cycles"]) for r in rows]
    two_gemm = [float(r["tileflow_2gemm_cycles"]) for r in rows]
    fused = [float(r["tileflow_fused_cycles"]) for r in rows]
    mape_two = mape(actual, two_gemm)
    mape_fused = mape(actual, fused)
    if mape_fused < mape_two:
        return "tileflow_fused_cycles", "fused", mape_fused
    return "tileflow_2gemm_cycles", "2gemm", mape_two


def build_rows(all_rows):
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["scenario_id"], []).append(row)

    final_rows = []
    choices = {}
    for scenario_id, rows in grouped.items():
        best_col, best_tag, best_mape = choose_best_tileflow(rows)
        choices[scenario_id] = {"column": best_col, "tag": best_tag, "mean_mape": best_mape}
        for row in rows:
            new_row = {
                "scenario_id": row["scenario_id"],
                "scenario_title": row["scenario_title"],
                "batch": row.get("batch", ""),
                "heads": row.get("heads", ""),
                "heads_q": row.get("heads_q", ""),
                "heads_kv": row.get("heads_kv", ""),
                "seq_q": row.get("seq_q", ""),
                "seq_kv": row.get("seq_kv", ""),
                "seq_len": row.get("seq_len", ""),
                "actual_cycles": row["actual_cycles"],
                "tiledataflow_sim_cycles": row["tiledataflow_sim_cycles"],
                "llmcompass_cycles": row["llmcompass_cycles"],
                "tileflow_cycles": row[best_col],
                "tileflow_variant": best_tag,
            }
            final_rows.append(new_row)
    return final_rows, choices


def write_csv(rows, out_path):
    fieldnames = [
        "scenario_id",
        "scenario_title",
        "batch",
        "heads",
        "heads_q",
        "heads_kv",
        "seq_q",
        "seq_kv",
        "seq_len",
        "actual_cycles",
        "tiledataflow_sim_cycles",
        "llmcompass_cycles",
        "tileflow_cycles",
        "tileflow_variant",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, choices, out_path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(row)

    fieldnames = ["scenario_id", "scenario_title", "tileflow_variant", "model", "mean_mape_percent", "pearson_corr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id, data in grouped.items():
            actual = [float(r["actual_cycles"]) for r in data]
            scenario_title = data[0]["scenario_title"]
            for col, label in [
                ("tiledataflow_sim_cycles", "TileDataflow"),
                ("llmcompass_cycles", "LLMCompass"),
                ("tileflow_cycles", "TileFlow"),
            ]:
                pred = [float(r[col]) for r in data]
                writer.writerow(
                    {
                        "scenario_id": scenario_id,
                        "scenario_title": scenario_title,
                        "tileflow_variant": choices[scenario_id]["tag"],
                        "model": label,
                        "mean_mape_percent": f"{mape(actual, pred):.6f}",
                        "pearson_corr": f"{corr(actual, pred):.6f}",
                    }
                )


def plot(rows, choices, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    scenario_order = ["fa2", "flashdecoding", "flashmla"]

    for ax, scenario_id in zip(axes[:3], scenario_order):
        data = grouped[scenario_id]
        data = sorted(data, key=lambda r: float(r["actual_cycles"]))
        x = list(range(1, len(data) + 1))
        actual = [float(r["actual_cycles"]) for r in data]
        tiledataflow = [float(r["tiledataflow_sim_cycles"]) for r in data]
        llmcompass = [float(r["llmcompass_cycles"]) for r in data]
        tileflow = [float(r["tileflow_cycles"]) for r in data]

        ax.plot(x, actual, color="#111827", linewidth=2.6, marker="o", markersize=3.2, label="Actual")
        ax.plot(x, tiledataflow, color=MODEL_COLORS["TileDataflow"], linewidth=1.8, marker="s", markersize=2.8, label="TileDataflow")
        ax.plot(x, llmcompass, color=MODEL_COLORS["LLMCompass"], linewidth=1.8, marker="^", markersize=2.8, label="LLMCompass")
        ax.plot(x, tileflow, color=MODEL_COLORS["TileFlow"], linewidth=1.8, marker="D", markersize=2.8, label=f"TileFlow ({choices[scenario_id]['tag']})")

        stat_text = "\n".join(
            [
                f"TileDataflow: {mape(actual, tiledataflow):.1f}% / r={corr(actual, tiledataflow):.3f}",
                f"LLMCompass: {mape(actual, llmcompass):.1f}% / r={corr(actual, llmcompass):.3f}",
                f"TileFlow-{choices[scenario_id]['tag']}: {mape(actual, tileflow):.1f}% / r={corr(actual, tileflow):.3f}",
            ]
        )
        ax.text(
            0.99,
            0.02,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
        ax.set_title(f"{data[0]['scenario_title']} (best TileFlow = {choices[scenario_id]['tag']})", fontsize=13)
        ax.set_xlabel("Workloads sorted by actual cycles")
        ax.set_ylabel("Cycles")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    ax = axes[3]
    xs = [0, 1, 2]
    width = 0.22
    td_vals = []
    llm_vals = []
    tf_vals = []
    labels = []
    for sid in scenario_order:
        data = grouped[sid]
        actual = [float(r["actual_cycles"]) for r in data]
        td_vals.append(mape(actual, [float(r["tiledataflow_sim_cycles"]) for r in data]))
        llm_vals.append(mape(actual, [float(r["llmcompass_cycles"]) for r in data]))
        tf_vals.append(mape(actual, [float(r["tileflow_cycles"]) for r in data]))
        labels.append(data[0]["scenario_title"])
    ax.bar([x - width for x in xs], td_vals, width=width, color=MODEL_COLORS["TileDataflow"], label="TileDataflow")
    ax.bar(xs, llm_vals, width=width, color=MODEL_COLORS["LLMCompass"], label="LLMCompass")
    ax.bar([x + width for x in xs], tf_vals, width=width, color=MODEL_COLORS["TileFlow"], label="TileFlow(best)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title("Best TileFlow Summary", fontsize=13)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle("Attention Family Comparison with Best TileFlow Result per Scenario", fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")


def main():
    source_csv = RESULT_DIR / "attention_family_all_models.csv"
    all_rows = read_rows(source_csv)
    rows, choices = build_rows(all_rows)

    combined_csv = RESULT_DIR / "attention_family_best_tileflow_comparison.csv"
    summary_csv = RESULT_DIR / "attention_family_best_tileflow_summary.csv"
    plot_path = RESULT_DIR / "attention_family_best_tileflow_comparison.png"

    write_csv(rows, combined_csv)
    write_summary(rows, choices, summary_csv)
    plot(rows, choices, plot_path)

    print("TileFlow choices:")
    for scenario_id, info in choices.items():
        print(f"  {scenario_id}: {info['tag']} (mean MAPE={info['mean_mape']:.2f}%)")
    print(f"Saved combined CSV: {combined_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
