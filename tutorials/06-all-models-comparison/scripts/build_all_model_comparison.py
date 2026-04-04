#!/usr/bin/env python3
import csv
import math
import os
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(TUTORIAL_DIR / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(TUTORIAL_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = TUTORIAL_DIR.parent.parent
LLM_REF_DIR = Path("/Users/jwhuang/Code/LLMCompass/ref")


SCENARIOS = [
    {
        "id": "a100_gemm",
        "title": "A100 GEMM",
        "hardware": "A100",
        "workload": "GEMM",
        "llm_csv": LLM_REF_DIR / "a100_gemm_llmcompass_comparison.csv",
        "tdf_csv": LLM_REF_DIR
        / "TileDataflowAnalyticalModel/Experiment/A100/GEMM/results/Multiple_SM_BMBNBK12825664_PERSISTENT/mnk_max_sm_span_512_2048_2stage_compare.csv",
        "tileflow_csv": REPO_ROOT
        / "tutorials/03-a100-gemm-persistent/results/mnk_max_sm_span_512_2048_2stage_compare.csv",
    },
    {
        "id": "h100_gemm",
        "title": "H100 GEMM",
        "hardware": "H100",
        "workload": "GEMM",
        "llm_csv": LLM_REF_DIR / "h100_gemm_llmcompass_comparison.csv",
        "tdf_csv": LLM_REF_DIR
        / "TileDataflowAnalyticalModel/Experiment/H100/GEMM/results/Multiple_SM_BMBNBK12825664_PERSISTENT/mnk_max_block_span_512_2048_2stage_compare.csv",
        "tileflow_csv": REPO_ROOT
        / "tutorials/02-h100-gemm-persistent/results/mnk_max_block_span_512_2048_2stage_compare.csv",
    },
    {
        "id": "h100_fused_gemm",
        "title": "H100 Fused GEMM + SiLU",
        "hardware": "H100",
        "workload": "Fused GEMM + SiLU",
        "llm_csv": LLM_REF_DIR / "h100_fused_gemm_llmcompass_comparison.csv",
        "tdf_csv": LLM_REF_DIR
        / "TileDataflowAnalyticalModel/Experiment/H100/Fused GEMM/results/mnk_max_block_span_512_2048_fused_silu_compare.csv",
        "tileflow_csv": REPO_ROOT
        / "tutorials/04-h100-fused-gemm/results/mnk_max_block_span_512_2048_fused_silu_compare.csv",
    },
    {
        "id": "h100_fp8_gemm",
        "title": "H100 FP8 GEMM",
        "hardware": "H100",
        "workload": "FP8 GEMM",
        "llm_csv": LLM_REF_DIR / "h100_fp8_gemm_llmcompass_comparison.csv",
        "tdf_csv": LLM_REF_DIR
        / "TileDataflowAnalyticalModel/Experiment/H100/Mixed Precision GEMM/results/fp8_bmbn128_overall_max_span_compare.csv",
        "tileflow_csv": REPO_ROOT
        / "tutorials/05-h100-fp8-gemm/results/fp8_bmbn128_overall_max_span_compare.csv",
    },
]

MODEL_COLUMNS = [
    ("tiledataflow_sim_cycles", "TileDataflow", "#2563eb", "s"),
    ("tileflow_sim_cycles", "TileFlow", "#ea580c", "^"),
    ("llmcompass_cycles", "LLMCompass", "#dc2626", "D"),
]

BAR_CHART_SCENARIOS = ["a100_gemm", "h100_fused_gemm", "h100_fp8_gemm"]


def workload_key(row):
    return int(row["M"]), int(row["N"]), int(row["K"])


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_tdf_rows(path):
    rows = {}
    for row in read_csv_rows(path):
        key = workload_key(row)
        rows[key] = {
            "M": key[0],
            "N": key[1],
            "K": key[2],
            "actual_cycles": int(float(row["actual_cycles"])),
            "tiledataflow_sim_cycles": int(float(row["sim_cycles"])),
        }
    return rows


def load_tileflow_rows(path):
    rows = {}
    for row in read_csv_rows(path):
        key = workload_key(row)
        rows[key] = {
            "M": key[0],
            "N": key[1],
            "K": key[2],
            "actual_cycles": int(float(row["actual_cycles"])),
            "tileflow_sim_cycles": int(float(row["sim_cycles"])),
        }
    return rows


def load_llm_rows(path):
    rows = {}
    for row in read_csv_rows(path):
        key = workload_key(row)
        rows[key] = {
            "M": key[0],
            "N": key[1],
            "K": key[2],
            "actual_cycles": int(float(row["actual_cycles"])),
            "tiledataflow_sim_cycles_from_llm_csv": int(float(row["tileflow_sim_cycles"])),
            "llmcompass_cycles": int(float(row["llmcompass_cycles"])),
        }
    return rows


def mape(actual, pred):
    total = 0.0
    count = 0
    for a, p in zip(actual, pred):
        if a == 0:
            continue
        total += abs(p - a) / a
        count += 1
    return 100.0 * total / count if count else 0.0


def pearson_corr(actual, pred):
    n = len(actual)
    if n == 0:
        return 0.0
    mean_a = sum(actual) / n
    mean_p = sum(pred) / n
    num = sum((a - mean_a) * (p - mean_p) for a, p in zip(actual, pred))
    den_a = sum((a - mean_a) ** 2 for a in actual)
    den_p = sum((p - mean_p) ** 2 for p in pred)
    den = math.sqrt(den_a * den_p)
    return num / den if den else 0.0


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def scenario_stats(rows):
    actual = [row["actual_cycles"] for row in rows]
    stats = {}
    for column, label, color, marker in MODEL_COLUMNS:
        pred = [row[column] for row in rows]
        errors = [100.0 * abs(p - a) / a for a, p in zip(actual, pred)]
        stats[label] = {
            "column": column,
            "color": color,
            "marker": marker,
            "mean_mape": sum(errors) / len(errors),
            "median_mape": median(errors),
            "max_mape": max(errors),
            "corr": pearson_corr(actual, pred),
        }
    return stats


def merge_scenario(config):
    tdf_rows = load_tdf_rows(config["tdf_csv"])
    tileflow_rows = load_tileflow_rows(config["tileflow_csv"])
    llm_rows = load_llm_rows(config["llm_csv"])

    common_keys = sorted(set(tdf_rows) & set(tileflow_rows) & set(llm_rows))
    if len(common_keys) != 64:
        raise RuntimeError(f"{config['id']} expected 64 workloads, got {len(common_keys)}")

    merged = []
    for key in common_keys:
        tdf_row = tdf_rows[key]
        tileflow_row = tileflow_rows[key]
        llm_row = llm_rows[key]

        if tdf_row["actual_cycles"] != tileflow_row["actual_cycles"]:
            raise RuntimeError(f"{config['id']} actual mismatch for {key}")
        if tdf_row["actual_cycles"] != llm_row["actual_cycles"]:
            raise RuntimeError(f"{config['id']} actual mismatch vs llm csv for {key}")
        if tdf_row["tiledataflow_sim_cycles"] != llm_row["tiledataflow_sim_cycles_from_llm_csv"]:
            raise RuntimeError(f"{config['id']} TileDataflow mismatch vs llm csv for {key}")

        actual_cycles = tdf_row["actual_cycles"]
        merged.append(
            {
                "scenario_id": config["id"],
                "scenario_title": config["title"],
                "hardware": config["hardware"],
                "workload": config["workload"],
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "actual_cycles": actual_cycles,
                "tiledataflow_sim_cycles": tdf_row["tiledataflow_sim_cycles"],
                "tileflow_sim_cycles": tileflow_row["tileflow_sim_cycles"],
                "llmcompass_cycles": llm_row["llmcompass_cycles"],
            }
        )

    return merged


def write_combined_csv(rows, out_path):
    fieldnames = [
        "scenario_id",
        "scenario_title",
        "hardware",
        "workload",
        "M",
        "N",
        "K",
        "actual_cycles",
        "tiledataflow_sim_cycles",
        "tileflow_sim_cycles",
        "llmcompass_cycles",
        "tiledataflow_mape_percent",
        "tileflow_mape_percent",
        "llmcompass_mape_percent",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            actual = row["actual_cycles"]
            writer.writerow(
                {
                    **row,
                    "tiledataflow_mape_percent": f"{100.0 * abs(row['tiledataflow_sim_cycles'] - actual) / actual:.6f}",
                    "tileflow_mape_percent": f"{100.0 * abs(row['tileflow_sim_cycles'] - actual) / actual:.6f}",
                    "llmcompass_mape_percent": f"{100.0 * abs(row['llmcompass_cycles'] - actual) / actual:.6f}",
                }
            )


def write_summary_csv(grouped_rows, out_path):
    fieldnames = ["scenario_id", "scenario_title", "model", "mean_mape_percent", "median_mape_percent", "max_mape_percent", "pearson_corr"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id, rows in grouped_rows.items():
            title = rows[0]["scenario_title"]
            stats = scenario_stats(rows)
            for label, values in stats.items():
                writer.writerow(
                    {
                        "scenario_id": scenario_id,
                        "scenario_title": title,
                        "model": label,
                        "mean_mape_percent": f"{values['mean_mape']:.6f}",
                        "median_mape_percent": f"{values['median_mape']:.6f}",
                        "max_mape_percent": f"{values['max_mape']:.6f}",
                        "pearson_corr": f"{values['corr']:.6f}",
                    }
                )


def plot_grouped_rows(grouped_rows, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (scenario_id, rows) in zip(axes, grouped_rows.items()):
        rows = sorted(rows, key=lambda row: (row["actual_cycles"], row["M"], row["N"], row["K"]))
        x = list(range(1, len(rows) + 1))
        actual = [row["actual_cycles"] for row in rows]
        labels = [f"{row['M']}x{row['N']}x{row['K']}" for row in rows]

        ax.plot(x, actual, color="#111827", linewidth=2.6, marker="o", markersize=3.5, label="Actual")
        for column, label, color, marker in MODEL_COLUMNS:
            pred = [row[column] for row in rows]
            ax.plot(x, pred, color=color, linewidth=1.8, marker=marker, markersize=3.2, alpha=0.95, label=label)

        stats_lines = []
        for column, label, _color, _marker in MODEL_COLUMNS:
            pred = [row[column] for row in rows]
            stats_lines.append(f"{label}: MAPE {mape(actual, pred):.1f}%, r={pearson_corr(actual, pred):.3f}")

        ax.set_title(rows[0]["scenario_title"], fontsize=14, pad=10)
        ax.set_xlabel("Workload index sorted by actual cycles")
        ax.set_ylabel("Cycles")
        ax.set_xticks(list(range(1, len(rows) + 1, 8)))
        ax.text(
            0.99,
            0.02,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
        ax.text(
            0.01,
            0.98,
            f"first={labels[0]}\nlast={labels[-1]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#e5e7eb"},
        )
        ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)

    fig.suptitle("Actual vs TileDataflow vs TileFlow vs LLMCompass Across All MNK Workloads", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")


def plot_paper_summary(grouped_rows, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)

    parity_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
    ]
    mape_ax = fig.add_subplot(gs[1, 1])
    corr_ax = fig.add_subplot(gs[1, 2])

    short_titles = {
        "a100_gemm": "A100 GEMM",
        "h100_gemm": "H100 GEMM",
        "h100_fused_gemm": "H100 Fused GEMM + SiLU",
        "h100_fp8_gemm": "H100 FP8 GEMM",
    }
    scenario_order = list(grouped_rows.keys())
    scenario_labels = [short_titles[k] for k in scenario_order]

    legend_handles = []
    legend_labels = []

    for ax, scenario_id in zip(parity_axes, scenario_order):
        rows = grouped_rows[scenario_id]
        actual = [row["actual_cycles"] for row in rows]
        scenario_max = max(
            max(actual),
            max(row["tiledataflow_sim_cycles"] for row in rows),
            max(row["tileflow_sim_cycles"] for row in rows),
            max(row["llmcompass_cycles"] for row in rows),
        )
        ax.plot([0, scenario_max], [0, scenario_max], linestyle="--", color="#6b7280", linewidth=1.2, zorder=1)

        stats = scenario_stats(rows)
        for column, label, color, marker in MODEL_COLUMNS:
            pred = [row[column] for row in rows]
            handle = ax.scatter(
                actual,
                pred,
                s=32,
                alpha=0.88,
                facecolors="none",
                edgecolors=color,
                linewidths=1.1,
                marker=marker,
                zorder=2,
            )
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

        stat_lines = [f"{label}: {stats[label]['mean_mape']:.1f}% / r={stats[label]['corr']:.3f}" for _col, label, _c, _m in MODEL_COLUMNS]
        ax.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
        )
        ax.set_title(short_titles[scenario_id], fontsize=12.5, pad=8)
        ax.set_xlabel("Actual cycles")
        ax.set_ylabel("Predicted cycles")
        ax.set_xlim(0, scenario_max * 1.03)
        ax.set_ylim(0, scenario_max * 1.03)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(4, 4))

    x = list(range(len(scenario_order)))
    width = 0.22

    for offset, (_column, label, color, _marker) in zip((-width, 0.0, width), MODEL_COLUMNS):
        values = [scenario_stats(grouped_rows[scenario_id])[label]["mean_mape"] for scenario_id in scenario_order]
        bars = mape_ax.bar([xi + offset for xi in x], values, width=width, color=color, alpha=0.92, label=label)
        for bar in bars:
            height = bar.get_height()
            mape_ax.text(bar.get_x() + bar.get_width() / 2, height + 0.7, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    mape_ax.set_title("Mean MAPE Across 64 Workloads", fontsize=12.5, pad=8)
    mape_ax.set_ylabel("MAPE (%)")
    mape_ax.set_xticks(x)
    mape_ax.set_xticklabels(["A100\nGEMM", "H100\nGEMM", "H100\nFused", "H100\nFP8"])
    mape_ax.legend(loc="upper left", fontsize=8.8, frameon=True)

    for offset, (_column, label, color, _marker) in zip((-width, 0.0, width), MODEL_COLUMNS):
        values = [scenario_stats(grouped_rows[scenario_id])[label]["corr"] for scenario_id in scenario_order]
        bars = corr_ax.bar([xi + offset for xi in x], values, width=width, color=color, alpha=0.92, label=label)
        for bar in bars:
            height = bar.get_height()
            corr_ax.text(bar.get_x() + bar.get_width() / 2, height + 0.006, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    corr_ax.set_title("Pearson Correlation", fontsize=12.5, pad=8)
    corr_ax.set_ylabel("Correlation")
    corr_ax.set_ylim(0.75, 1.04)
    corr_ax.set_xticks(x)
    corr_ax.set_xticklabels(["A100\nGEMM", "H100\nGEMM", "H100\nFused", "H100\nFP8"])

    fig.legend(
        [plt.Line2D([0], [0], linestyle="--", color="#6b7280", linewidth=1.2)] + legend_handles,
        ["Ideal"] + legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10.5,
    )
    fig.suptitle("Actual vs TileDataflow vs TileFlow vs LLMCompass", fontsize=18, y=1.03)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.07, right=0.98, hspace=0.34, wspace=0.28)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")


def plot_scenario_bar_chart(rows, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    rows = sorted(rows, key=lambda row: (row["actual_cycles"], row["M"], row["N"], row["K"]))
    x = list(range(len(rows)))
    actual = [row["actual_cycles"] for row in rows]
    width = 0.2

    fig, ax = plt.subplots(figsize=(19, 6.5))
    actual_bars = ax.bar([v - 1.5 * width for v in x], actual, width=width, color="#111827", alpha=0.95, label="Actual")
    model_bars = []
    for offset, (column, label, color, _marker) in zip((-0.5 * width, 0.5 * width, 1.5 * width), MODEL_COLUMNS):
        pred = [row[column] for row in rows]
        bars = ax.bar([v + offset for v in x], pred, width=width, color=color, alpha=0.9, label=label)
        model_bars.append((label, pred, bars))

    tick_step = 4
    tick_positions = list(range(0, len(rows), tick_step))
    tick_labels = [f"{rows[i]['M']}x{rows[i]['N']}x{rows[i]['K']}" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Workloads sorted by actual cycles")
    ax.set_ylabel("Cycles")
    ax.set_title(f"{rows[0]['scenario_title']}: Actual vs TileDataflow vs TileFlow vs LLMCompass", fontsize=15, pad=12)
    ax.legend(loc="upper left", ncol=4, frameon=True)

    stats = scenario_stats(rows)
    stats_text = "\n".join(
        f"{label}: MAPE {stats[label]['mean_mape']:.1f}%, r={stats[label]['corr']:.3f}"
        for _column, label, _color, _marker in MODEL_COLUMNS
    )
    ax.text(
        0.995,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.96},
    )

    ymax = max(
        max(actual),
        max(max(pred) for _label, pred, _bars in model_bars),
    )
    ax.set_ylim(0, ymax * 1.16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    result_dir = TUTORIAL_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    grouped_rows = {}
    for config in SCENARIOS:
        rows = merge_scenario(config)
        grouped_rows[config["id"]] = rows
        all_rows.extend(rows)

    combined_csv = result_dir / "all_model_cycle_comparison.csv"
    summary_csv = result_dir / "all_model_error_summary.csv"
    plot_path = result_dir / "all_model_cycle_comparison.png"
    paper_plot_path = result_dir / "all_model_cycle_comparison_paper.png"
    bar_dir = result_dir / "bar_charts"

    write_combined_csv(all_rows, combined_csv)
    write_summary_csv(grouped_rows, summary_csv)
    plot_grouped_rows(grouped_rows, plot_path)
    plot_paper_summary(grouped_rows, paper_plot_path)
    bar_dir.mkdir(parents=True, exist_ok=True)
    for scenario_id in BAR_CHART_SCENARIOS:
        plot_scenario_bar_chart(grouped_rows[scenario_id], bar_dir / f"{scenario_id}_bar.png")

    print(f"Saved combined CSV: {combined_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved paper plot: {paper_plot_path}")
    print(f"Saved bar charts under: {bar_dir}")


if __name__ == "__main__":
    main()
