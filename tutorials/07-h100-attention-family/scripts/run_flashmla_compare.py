#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

from common import TUTORIAL_DIR, default_tileflow_path, run_gemm_case


DEFAULT_ACTUAL_CSV = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/"
    "H100/FlashMLA/results/flash_mla_validation_vs_sim.csv"
)
DEFAULT_OUT_CSV = TUTORIAL_DIR / "results" / "flash_mla_validation_vs_sim_tileflow.csv"
QK_DIM = 576
PV_DIM = 512


def read_actual(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["batch"]),
                    int(row["heads"]),
                    int(row["seq_len"]),
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
    raw_dir = out_path.parent / "flashmla_raw"
    macro_dir = out_path.parent / "flashmla_macro"
    raw_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(actual_rows)
    for idx, (batch, heads, seq_len, actual_cycles) in enumerate(actual_rows, start=1):
        m = batch * heads
        qk_cycles = run_gemm_case(args.tileflow, raw_dir, macro_dir, f"flashmla_{idx:02d}_qk", m, seq_len, QK_DIM)
        pv_cycles = run_gemm_case(args.tileflow, raw_dir, macro_dir, f"flashmla_{idx:02d}_pv", m, PV_DIM, seq_len)
        sim_cycles = qk_cycles + pv_cycles
        mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100.0 if actual_cycles else 0.0
        print(
            f"[{idx:02d}/{total}] batch={batch} heads={heads} seq={seq_len} "
            f"actual={actual_cycles} tileflow={sim_cycles} qk={qk_cycles} pv={pv_cycles} mape={mape:.2f}%"
        )
        results.append((batch, heads, seq_len, actual_cycles, sim_cycles, f"{mape:.6f}"))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "heads", "seq_len", "actual_cycles", "sim_cycles", "mape_percent"])
        writer.writerows(results)

    print(f"Saved CSV: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
