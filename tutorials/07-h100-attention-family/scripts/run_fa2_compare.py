#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

from common import TUTORIAL_DIR, default_tileflow_path, run_gemm_case


DEFAULT_ACTUAL_CSV = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/"
    "H100/FA2/results/fa2_pipeline_training_overall_max_span_compare.csv"
)
DEFAULT_OUT_CSV = TUTORIAL_DIR / "results" / "fa2_pipeline_training_overall_max_span_compare_tileflow.csv"
HEAD_DIM = 128


def read_actual(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["batch"]),
                    int(row["heads_q"]),
                    int(row["heads_kv"]),
                    int(row["seq_q"]),
                    int(row["seq_kv"]),
                    int(float(row["actual_cycles"])),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow", default=default_tileflow_path())
    parser.add_argument("--actual-csv", default=str(DEFAULT_ACTUAL_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    args = parser.parse_args()

    actual_rows = read_actual(args.actual_csv)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / "fa2_raw"
    macro_dir = out_path.parent / "fa2_macro"
    raw_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(actual_rows)
    for idx, (batch, heads_q, heads_kv, seq_q, seq_kv, actual_cycles) in enumerate(actual_rows, start=1):
        m = batch * heads_q * seq_q
        qk_cycles = run_gemm_case(args.tileflow, raw_dir, macro_dir, f"fa2_{idx:02d}_qk", m, seq_kv, HEAD_DIM)
        pv_cycles = run_gemm_case(args.tileflow, raw_dir, macro_dir, f"fa2_{idx:02d}_pv", m, HEAD_DIM, seq_kv)
        sim_cycles = qk_cycles + pv_cycles
        mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100.0 if actual_cycles else 0.0
        print(
            f"[{idx:02d}/{total}] batch={batch} heads={heads_q}/{heads_kv} seq={seq_q}/{seq_kv} "
            f"actual={actual_cycles} tileflow={sim_cycles} qk={qk_cycles} pv={pv_cycles} mape={mape:.2f}%"
        )
        results.append((batch, heads_q, heads_kv, seq_q, seq_kv, actual_cycles, sim_cycles))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "heads_q", "heads_kv", "seq_q", "seq_kv", "actual_cycles", "sim_cycles"])
        writer.writerows(results)

    print(f"Saved CSV: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
