#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
RESULT_DIR = TUTORIAL_DIR / "results"

SCENARIO_META = {
    "fa2": {"scenario_title": "FA2", "hardware": "H100", "workload": "FA2"},
    "flashdecoding": {"scenario_title": "FlashDecoding", "hardware": "H100", "workload": "FlashDecoding"},
    "flashmla": {"scenario_title": "FlashMLA", "hardware": "H100", "workload": "FlashMLA"},
}


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mape(actual, pred):
    actual_f = float(actual)
    pred_f = float(pred)
    return 100.0 * abs(pred_f - actual_f) / actual_f if actual_f else 0.0


def build_rows(source_rows, tileflow_column):
    out = []
    for row in source_rows:
        scenario_id = row["scenario_id"]
        meta = SCENARIO_META[scenario_id]
        out.append(
            {
                "scenario_id": scenario_id,
                "scenario_title": meta["scenario_title"],
                "hardware": meta["hardware"],
                "workload": meta["workload"],
                "batch": row.get("batch", ""),
                "heads": row.get("heads", ""),
                "heads_q": row.get("heads_q", ""),
                "heads_kv": row.get("heads_kv", ""),
                "seq_q": row.get("seq_q", ""),
                "seq_kv": row.get("seq_kv", ""),
                "seq_len": row.get("seq_len", ""),
                "actual_cycles": row["actual_cycles"],
                "tiledataflow_sim_cycles": row["tiledataflow_sim_cycles"],
                "tileflow_sim_cycles": row[tileflow_column],
                "llmcompass_cycles": row["llmcompass_cycles"],
                "tiledataflow_mape_percent": f"{mape(row['actual_cycles'], row['tiledataflow_sim_cycles']):.6f}",
                "tileflow_mape_percent": f"{mape(row['actual_cycles'], row[tileflow_column]):.6f}",
                "llmcompass_mape_percent": f"{mape(row['actual_cycles'], row['llmcompass_cycles']):.6f}",
            }
        )
    return out


def write_rows(rows, out_path):
    fieldnames = [
        "scenario_id",
        "scenario_title",
        "hardware",
        "workload",
        "batch",
        "heads",
        "heads_q",
        "heads_kv",
        "seq_q",
        "seq_kv",
        "seq_len",
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
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_csv",
        default=str(RESULT_DIR / "attention_family_all_models.csv"),
        help="Source CSV produced by build_attention_family_comparison.py",
    )
    parser.add_argument(
        "--tileflow-column",
        default="tileflow_fused_cycles",
        choices=["tileflow_2gemm_cycles", "tileflow_fused_cycles"],
        help="Which TileFlow result to export as tileflow_sim_cycles",
    )
    parser.add_argument(
        "--out",
        default=str(RESULT_DIR / "attention_all_models_fused.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    source_rows = read_rows(Path(args.input_csv))
    rows = build_rows(source_rows, args.tileflow_column)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(rows, out_path)
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()
