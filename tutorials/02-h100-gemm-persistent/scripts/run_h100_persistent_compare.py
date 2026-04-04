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
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/"
    "Experiment/H100/GEMM/Validation Data/"
    "Multiple_SM_BMBNBK12825664_PERSISTENT/"
    "mnk_max_block_span_512_2048_2stage.csv"
)
DEFAULT_OUT_CSV = TUTORIAL_DIR / "results" / "mnk_max_block_span_512_2048_2stage_compare.csv"

BM = 128
BN = 256
BK = 64


def default_tileflow_path():
    for candidate in (
        REPO_ROOT / "build" / "bin" / "tileflow",
        REPO_ROOT / "bin" / "tileflow",
    ):
        if candidate.exists():
            return str(candidate)
    return "tileflow"


def read_actual(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["M"]),
                    int(row["N"]),
                    int(row["K"]),
                    int(float(row["max_block_span"])),
                )
            )
    return rows


def make_macro_yaml(path, m, n, k, output_prefix):
    if m % BM != 0 or n % BN != 0 or k % BK != 0:
        raise ValueError(
            f"M/N/K must be divisible by {BM}/{BN}/{BK}, got {m}/{n}/{k}"
        )

    mo = m // BM
    no = n // BN
    ko = k // BK

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
  MO: {mo}
  NO: {no}
  KO: {ko}
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
    dyld_paths = [
        str(REPO_ROOT / "build" / "lib"),
        str(REPO_ROOT / "3rdparty" / "timeloop" / "lib"),
    ]
    old_dyld = env.get("DYLD_LIBRARY_PATH")
    if old_dyld:
        dyld_paths.append(old_dyld)
    env["DYLD_LIBRARY_PATH"] = ":".join(dyld_paths)

    cmd = [
        tileflow_bin,
        str(TUTORIAL_DIR / "arch" / "arch.yaml"),
        str(TUTORIAL_DIR / "prob" / "prob.yaml"),
        str(TUTORIAL_DIR / "map" / "map.yaml"),
        str(macro_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow", default=default_tileflow_path(), help="TileFlow binary")
    parser.add_argument("--actual-csv", default=str(DEFAULT_ACTUAL_CSV), help="Actual CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV), help="Output compare CSV")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep per-case TileFlow csv outputs and macro files",
    )
    args = parser.parse_args()

    actual_rows = read_actual(args.actual_csv)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = out_path.parent / "raw"
    macro_dir = out_path.parent / "macro"
    raw_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(actual_rows)
    for idx, (m, n, k, actual_cycles) in enumerate(actual_rows, start=1):
        stem = f"M{m}_N{n}_K{k}"
        macro_path = macro_dir / f"{stem}.yaml"
        output_prefix = raw_dir / stem
        output_csv = output_prefix.with_suffix(".csv")

        make_macro_yaml(macro_path, m, n, k, output_prefix)
        run_tileflow(args.tileflow, macro_path)
        sim_cycles = read_cycle(output_csv)

        err = abs(sim_cycles - actual_cycles) / actual_cycles * 100 if actual_cycles else 0.0
        print(
            f"[{idx:02d}/{total}] M={m} N={n} K={k} "
            f"actual={actual_cycles} tileflow={sim_cycles} mape={err:.2f}%"
        )

        results.append((m, n, k, 2, actual_cycles, sim_cycles))

        if not args.keep_raw:
            if macro_path.exists():
                macro_path.unlink()
            if output_csv.exists():
                output_csv.unlink()
            mapping_txt = output_prefix.with_suffix(".mapping.txt")
            if mapping_txt.exists():
                mapping_txt.unlink()
            tuning_csv = output_prefix.with_suffix(".tuning.csv")
            if tuning_csv.exists():
                tuning_csv.unlink()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "stage", "actual_cycles", "sim_cycles"])
        writer.writerows(results)

    if not args.keep_raw:
        if macro_dir.exists() and not any(macro_dir.iterdir()):
            macro_dir.rmdir()
        if raw_dir.exists() and not any(raw_dir.iterdir()):
            raw_dir.rmdir()

    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
