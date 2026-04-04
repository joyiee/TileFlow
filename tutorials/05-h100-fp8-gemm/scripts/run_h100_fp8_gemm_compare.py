#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
REPO_ROOT = TUTORIAL_DIR.parent.parent

DEFAULT_ACTUAL_CSV = Path(
    "/Users/jwhuang/Code/simulator/TileDataflowAnalyticalModel/Experiment/"
    "H100/Mixed Precision GEMM/Validation Data/fp8_bmbn128_overall_max_span_512_2048.csv"
)
DEFAULT_OUT_CSV = TUTORIAL_DIR / "results" / "fp8_bmbn128_overall_max_span_compare.csv"

BM = 128
BN = 128
BK = 64


def default_tileflow_path():
    for candidate in (REPO_ROOT / "build" / "bin" / "tileflow", REPO_ROOT / "bin" / "tileflow"):
        if candidate.exists():
            return str(candidate)
    return "tileflow"


def read_actual(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["M"]), int(row["N"]), int(row["K"]), int(float(row["overall_max_span_cycles"]))))
    return rows


def make_macro_yaml(path, m, n, k, output_prefix):
    text = f"""output: {output_prefix}
verbose: 0
check:
  mem: True
  loopcount: True
  spatial: True
macro:
  M: {m}
  N: {n}
  K: {k}
  MO: {m // BM}
  NO: {n // BN}
  KO: {k // BK}
"""
    path.write_text(text)


def read_cycle(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == "Cycle":
                return int(float(row[1]))
    raise RuntimeError(f"Cycle not found in {csv_path}")


def run_tileflow(tileflow_bin, macro_path):
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = ":".join(
        [
            str(REPO_ROOT / "build" / "lib"),
            str(REPO_ROOT / "3rdparty" / "timeloop" / "lib"),
            env.get("DYLD_LIBRARY_PATH", ""),
        ]
    ).strip(":")
    cmd = [
        tileflow_bin,
        str(TUTORIAL_DIR / "arch" / "arch.yaml"),
        str(TUTORIAL_DIR / "prob" / "prob.yaml"),
        str(TUTORIAL_DIR / "map" / "map.yaml"),
        str(macro_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow", default=default_tileflow_path())
    parser.add_argument("--actual-csv", default=str(DEFAULT_ACTUAL_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    args = parser.parse_args()

    actual_rows = read_actual(args.actual_csv)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / "raw"
    macro_dir = out_path.parent / "macro"
    raw_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (m, n, k, actual_cycles) in enumerate(actual_rows, start=1):
        stem = f"M{m}_N{n}_K{k}"
        macro_path = macro_dir / f"{stem}.yaml"
        output_prefix = raw_dir / stem
        output_csv = output_prefix.with_suffix(".csv")
        make_macro_yaml(macro_path, m, n, k, output_prefix)
        run_tileflow(args.tileflow, macro_path)
        sim_cycles = read_cycle(output_csv)
        mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100 if actual_cycles else 0.0
        print(f"[{idx:02d}/{len(actual_rows)}] M={m} N={n} K={k} actual={actual_cycles} tileflow={sim_cycles} mape={mape:.2f}%")
        results.append((m, n, k, actual_cycles, sim_cycles, f"{mape:.6f}"))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "actual_cycles", "sim_cycles", "mape_percent"])
        writer.writerows(results)

    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
