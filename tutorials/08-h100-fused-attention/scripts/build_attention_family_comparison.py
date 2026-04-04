#!/usr/bin/env python3
import csv
import math
import os
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(TUTORIAL_DIR / "results" / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(TUTORIAL_DIR / "results" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LLM_REF = Path("/Users/jwhuang/Code/LLMCompass/ref")
REPO_ROOT = TUTORIAL_DIR.parent.parent

SCENARIOS = [
    {
        "id": "fa2",
        "title": "FA2",
        "tdf_csv": LLM_REF / "TileDataflowAnalyticalModel/Experiment/H100/FA2/results/fa2_pipeline_training_overall_max_span_compare.csv",
        "llm_csv": LLM_REF / "attention_results/fa2_comparison.csv",
        "tileflow_2gemm_csv": REPO_ROOT / "tutorials/07-h100-attention-family/results/fa2_pipeline_training_overall_max_span_compare_tileflow.csv",
        "tileflow_fused_csv": TUTORIAL_DIR / "results/fa2_fused_compare_tileflow.csv",
        "keys": ["batch", "heads_q", "heads_kv", "seq_q", "seq_kv"],
    },
    {
        "id": "flashdecoding",
        "title": "FlashDecoding",
        "tdf_csv": LLM_REF / "TileDataflowAnalyticalModel/Experiment/H100/FlashDecoding/results/flashdecoding_overall_max_span_compare.csv",
        "llm_csv": LLM_REF / "attention_results/flashdecoding_comparison.csv",
        "tileflow_2gemm_csv": REPO_ROOT / "tutorials/07-h100-attention-family/results/flashdecoding_overall_max_span_compare_tileflow.csv",
        "tileflow_fused_csv": TUTORIAL_DIR / "results/flashdecoding_fused_compare_tileflow.csv",
        "keys": ["batch", "heads_q", "heads_kv", "seq_q", "seq_kv"],
    },
    {
        "id": "flashmla",
        "title": "FlashMLA",
        "tdf_csv": LLM_REF / "TileDataflowAnalyticalModel/Experiment/H100/FlashMLA/results/flash_mla_validation_vs_sim.csv",
        "llm_csv": LLM_REF / "attention_results/flashmla_comparison.csv",
        "tileflow_2gemm_csv": REPO_ROOT / "tutorials/07-h100-attention-family/results/flash_mla_validation_vs_sim_tileflow.csv",
        "tileflow_fused_csv": TUTORIAL_DIR / "results/flashmla_fused_compare_tileflow.csv",
        "keys": ["batch", "heads", "seq_len"],
    },
]

MODELS = [
    ("tiledataflow_sim_cycles", "TileDataflow", "#2563eb"),
    ("llmcompass_cycles", "LLMCompass", "#dc2626"),
    ("tileflow_2gemm_cycles", "TileFlow-2GEMM", "#ea580c"),
    ("tileflow_fused_cycles", "TileFlow-Fused", "#059669"),
]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_key(row, key_names):
    return tuple(int(row[k]) for k in key_names)


def load_tdf(path, key_names):
    out = {}
    for row in read_rows(path):
        key = make_key(row, key_names)
        out[key] = {
            "actual_cycles": int(float(row["actual_cycles"])),
            "tiledataflow_sim_cycles": int(float(row["sim_cycles"])),
        }
    return out


def load_llm(path, key_names):
    out = {}
    for row in read_rows(path):
        key = make_key(row, key_names)
        out[key] = {
            "actual_cycles": int(float(row["actual_cycles"])),
            "tiledataflow_sim_cycles_from_llm": int(float(row["tileflow_sim_cycles"])),
            "llmcompass_cycles": int(float(row["llmcompass_cycles"])),
        }
    return out


def load_tileflow(path, key_names):
    out = {}
    for row in read_rows(path):
        key = make_key(row, key_names)
        out[key] = {
            "actual_cycles": int(float(row["actual_cycles"])),
            "sim_cycles": int(float(row["sim_cycles"])),
        }
    return out


def mape(actual, pred):
    return sum(abs(p - a) / a for a, p in zip(actual, pred) if a) * 100.0 / len(actual)


def corr(actual, pred):
    n = len(actual)
    mean_a = sum(actual) / n
    mean_p = sum(pred) / n
    cov = sum((a - mean_a) * (p - mean_p) for a, p in zip(actual, pred))
    var_a = sum((a - mean_a) ** 2 for a in actual)
    var_p = sum((p - mean_p) ** 2 for p in pred)
    den = math.sqrt(var_a * var_p)
    return cov / den if den else 0.0


def median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def merge_scenario(config):
    tdf = load_tdf(config["tdf_csv"], config["keys"])
    llm = load_llm(config["llm_csv"], config["keys"])
    tf_2g = load_tileflow(config["tileflow_2gemm_csv"], config["keys"])
    tf_fused = load_tileflow(config["tileflow_fused_csv"], config["keys"])

    keys = sorted(set(tdf) & set(llm) & set(tf_2g) & set(tf_fused))
    rows = []
    for key in keys:
        if tdf[key]["actual_cycles"] != llm[key]["actual_cycles"]:
            raise RuntimeError(f"{config['id']} actual mismatch for {key}")
        if tdf[key]["tiledataflow_sim_cycles"] != llm[key]["tiledataflow_sim_cycles_from_llm"]:
            raise RuntimeError(f"{config['id']} TileDataflow mismatch for {key}")
        row = {
            "scenario_id": config["id"],
            "scenario_title": config["title"],
            "actual_cycles": tdf[key]["actual_cycles"],
            "tiledataflow_sim_cycles": tdf[key]["tiledataflow_sim_cycles"],
            "llmcompass_cycles": llm[key]["llmcompass_cycles"],
            "tileflow_2gemm_cycles": tf_2g[key]["sim_cycles"],
            "tileflow_fused_cycles": tf_fused[key]["sim_cycles"],
        }
        for name, value in zip(config["keys"], key):
            row[name] = value
        rows.append(row)
    return rows


def write_combined(rows, out_path):
    key_fields = sorted({k for row in rows for k in row.keys()} - {"actual_cycles", "tiledataflow_sim_cycles", "llmcompass_cycles", "tileflow_2gemm_cycles", "tileflow_fused_cycles"})
    fieldnames = key_fields + [
        "actual_cycles",
        "tiledataflow_sim_cycles",
        "llmcompass_cycles",
        "tileflow_2gemm_cycles",
        "tileflow_fused_cycles",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(grouped_rows, out_path):
    fieldnames = ["scenario_id", "scenario_title", "model", "mean_mape_percent", "median_mape_percent", "max_mape_percent", "pearson_corr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id, rows in grouped_rows.items():
            actual = [r["actual_cycles"] for r in rows]
            for column, label, _color in MODELS:
                pred = [r[column] for r in rows]
                errors = [abs(p - a) / a * 100.0 for a, p in zip(actual, pred)]
                writer.writerow(
                    {
                        "scenario_id": scenario_id,
                        "scenario_title": rows[0]["scenario_title"],
                        "model": label,
                        "mean_mape_percent": f"{sum(errors) / len(errors):.6f}",
                        "median_mape_percent": f"{median(errors):.6f}",
                        "max_mape_percent": f"{max(errors):.6f}",
                        "pearson_corr": f"{corr(actual, pred):.6f}",
                    }
                )


def plot(grouped_rows, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes = axes.flatten()

    scenario_ids = list(grouped_rows.keys())
    for ax, scenario_id in zip(axes[:3], scenario_ids):
        rows = sorted(grouped_rows[scenario_id], key=lambda r: (r["actual_cycles"],) + tuple(v for k, v in r.items() if k.endswith("q") or k.endswith("kv") or k.endswith("len")))
        x = list(range(1, len(rows) + 1))
        actual = [r["actual_cycles"] for r in rows]
        ax.plot(x, actual, color="#111827", linewidth=2.5, marker="o", markersize=3, label="Actual")
        stat_lines = []
        for column, label, color in MODELS:
            pred = [r[column] for r in rows]
            ax.plot(x, pred, color=color, linewidth=1.8, marker="s", markersize=2.8, label=label)
            stat_lines.append(f"{label}: {mape(actual, pred):.1f}% / r={corr(actual, pred):.3f}")
        ax.set_title(rows[0]["scenario_title"], fontsize=13)
        ax.set_xlabel("Workloads sorted by actual cycles")
        ax.set_ylabel("Cycles")
        ax.text(
            0.99,
            0.02,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
        ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=True)

    summary_ax = axes[3]
    xs = list(range(len(scenario_ids)))
    width = 0.18
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, (column, label, color) in zip(offsets, MODELS):
        vals = []
        for scenario_id in scenario_ids:
            rows = grouped_rows[scenario_id]
            actual = [r["actual_cycles"] for r in rows]
            pred = [r[column] for r in rows]
            vals.append(mape(actual, pred))
        bars = summary_ax.bar([x + offset for x in xs], vals, width=width, color=color, alpha=0.92, label=label)
        for bar in bars:
            h = bar.get_height()
            summary_ax.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8)
    summary_ax.set_title("Mean MAPE Summary", fontsize=13)
    summary_ax.set_ylabel("MAPE (%)")
    summary_ax.set_xticks(xs)
    summary_ax.set_xticklabels([grouped_rows[sid][0]["scenario_title"] for sid in scenario_ids])
    summary_ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle("Attention Family: Actual vs TileDataflow vs LLMCompass vs TileFlow", fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")


def main():
    result_dir = TUTORIAL_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows = {}
    all_rows = []
    for config in SCENARIOS:
        rows = merge_scenario(config)
        grouped_rows[config["id"]] = rows
        all_rows.extend(rows)

    combined_csv = result_dir / "attention_family_all_models.csv"
    summary_csv = result_dir / "attention_family_error_summary.csv"
    plot_path = result_dir / "attention_family_all_models.png"

    write_combined(all_rows, combined_csv)
    write_summary(grouped_rows, summary_csv)
    plot(grouped_rows, plot_path)

    print(f"Saved combined CSV: {combined_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
