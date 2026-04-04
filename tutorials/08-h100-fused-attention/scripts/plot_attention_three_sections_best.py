#!/usr/bin/env python3
import argparse
import csv
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


def load_fa2(path):
    rows = list(csv.DictReader(open(path, newline="")))
    groups = {}
    tileflow_variant = None
    for r in rows:
        if r["scenario_id"] != "fa2":
            continue
        hq = int(r["heads_q"])
        hk = int(r["heads_kv"])
        sq = int(r["seq_q"])
        sk = int(r["seq_kv"])
        tileflow_variant = r["tileflow_variant"]
        groups.setdefault((hq, hk), []).append(
            {
                "x_label": f"{sk}",
                "actual": float(r["actual_cycles"]),
                "tiledataflow": float(r["tiledataflow_sim_cycles"]),
                "llmcompass": float(r["llmcompass_cycles"]),
                "tileflow": float(r["tileflow_cycles"]),
                "sort_key": (sq, sk),
            }
        )
    out = []
    for (hq, hk) in sorted(groups.keys()):
        vals = sorted(groups[(hq, hk)], key=lambda x: x["sort_key"])
        out.append({"group_label": f"Head={hq}/{hk}", "bars": vals})
    note = f"training: batch=1, seq_q=seq_kv, TileFlow={tileflow_variant}"
    return out, note


def load_flashdecoding(path):
    rows = list(csv.DictReader(open(path, newline="")))
    groups = {}
    tileflow_variant = None
    hq = hk = sq = None
    for r in rows:
        if r["scenario_id"] != "flashdecoding":
            continue
        b = int(r["batch"])
        hq = int(r["heads_q"])
        hk = int(r["heads_kv"])
        sq = int(r["seq_q"])
        sk = int(r["seq_kv"])
        tileflow_variant = r["tileflow_variant"]
        groups.setdefault(b, []).append(
            {
                "x_label": f"{sk}",
                "actual": float(r["actual_cycles"]),
                "tiledataflow": float(r["tiledataflow_sim_cycles"]),
                "llmcompass": float(r["llmcompass_cycles"]),
                "tileflow": float(r["tileflow_cycles"]),
                "sort_key": sk,
            }
        )
    out = []
    for b in sorted(groups.keys()):
        vals = sorted(groups[b], key=lambda x: x["sort_key"])
        out.append({"group_label": f"Batch={b}", "bars": vals})
    note = f"inference: heads={hq}/{hk}, seq_q={sq}, TileFlow={tileflow_variant}"
    return out, note


def load_flashmla(path):
    rows = list(csv.DictReader(open(path, newline="")))
    groups = {}
    tileflow_variant = None
    heads = None
    for r in rows:
        if r["scenario_id"] != "flashmla":
            continue
        b = int(r["batch"])
        heads = int(r["heads"])
        sk = int(r["seq_len"])
        tileflow_variant = r["tileflow_variant"]
        groups.setdefault(b, []).append(
            {
                "x_label": f"{sk}",
                "actual": float(r["actual_cycles"]),
                "tiledataflow": float(r["tiledataflow_sim_cycles"]),
                "llmcompass": float(r["llmcompass_cycles"]),
                "tileflow": float(r["tileflow_cycles"]),
                "sort_key": sk,
            }
        )
    out = []
    for b in sorted(groups.keys()):
        vals = sorted(groups[b], key=lambda x: x["sort_key"])
        out.append({"group_label": f"Batch={b}", "bars": vals})
    note = f"DeepSeek V3: heads={heads}, TileFlow={tileflow_variant}"
    return out, note


def pack_section(section_groups, start_x, group_gap=1.1, case_gap=1.0):
    group_centers = []
    case_centers = []
    x = start_x
    for g in section_groups:
        g_start = x
        for bar in g["bars"]:
            case_centers.append((x, bar["x_label"], bar))
            x += case_gap
        g_end = x - case_gap
        group_centers.append(((g_start + g_end) / 2.0, g["group_label"]))
        x += group_gap
    end_x = x - group_gap
    return {"case_centers": case_centers, "group_centers": group_centers, "start": start_x, "end": end_x}, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(RESULT_DIR / "attention_family_best_tileflow_comparison.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(RESULT_DIR / "attention_family_best_tileflow_three_sections.png"),
    )
    args = parser.parse_args()

    fa2_groups, fa2_note = load_fa2(args.csv)
    fd_groups, fd_note = load_flashdecoding(args.csv)
    mla_groups, mla_note = load_flashmla(args.csv)

    sec_gap = 1.8
    sec1, nx = pack_section(fa2_groups, 0.0)
    sec2, nx = pack_section(fd_groups, nx + sec_gap)
    sec3, _ = pack_section(mla_groups, nx + sec_gap)

    all_cases = sec1["case_centers"] + sec2["case_centers"] + sec3["case_centers"]
    xs = [x for x, _label, _bar in all_cases]
    labels = [label for _x, label, _bar in all_cases]
    actual = [bar["actual"] for _x, _label, bar in all_cases]
    tiledataflow = [bar["tiledataflow"] for _x, _label, bar in all_cases]
    llmcompass = [bar["llmcompass"] for _x, _label, bar in all_cases]
    tileflow = [bar["tileflow"] for _x, _label, bar in all_cases]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(22, 7.5))

    width = 0.18
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    ax.bar([x + offsets[0] for x in xs], actual, width=width, color="#111827", label="Actual")
    ax.bar([x + offsets[1] for x in xs], tiledataflow, width=width, color="#2563eb", label="TileDataflow")
    ax.bar([x + offsets[2] for x in xs], llmcompass, width=width, color="#dc2626", label="LLMCompass")
    ax.bar([x + offsets[3] for x in xs], tileflow, width=width, color="#059669", label="TileFlow(best)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Cycles", fontsize=12)
    ax.grid(axis="y", alpha=0.25)

    sep12 = (sec1["end"] + sec2["start"]) / 2.0
    sep23 = (sec2["end"] + sec3["start"]) / 2.0
    ax.axvline(sep12, color="#BDBDBD", linewidth=1.0)
    ax.axvline(sep23, color="#BDBDBD", linewidth=1.0)

    for cx, gl in sec1["group_centers"] + sec2["group_centers"] + sec3["group_centers"]:
        ax.text(cx, -0.12, gl, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    c1 = (sec1["start"] + sec1["end"]) / 2.0
    c2 = (sec2["start"] + sec2["end"]) / 2.0
    c3 = (sec3["start"] + sec3["end"]) / 2.0
    ax.text(c1, -0.20, "Flash Attention", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
    ax.text(c2, -0.20, "Flash Decoding", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
    ax.text(c3, -0.20, "FlashMLA", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)

    ax.text(c1, -0.28, fa2_note, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.text(c2, -0.28, fd_note, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.text(c3, -0.28, mla_note, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)

    ax.legend(loc="upper left", ncol=4, frameon=True)
    plt.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.36)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.0)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
