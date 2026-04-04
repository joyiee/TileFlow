#!/usr/bin/env python3
import csv
import os
import subprocess
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = THIS_DIR.parent
REPO_ROOT = TUTORIAL_DIR.parent.parent

M_TILE = 2048
N_TILE = 128
K_TILE = 64


def default_tileflow_path():
    for candidate in (REPO_ROOT / "build" / "bin" / "tileflow", REPO_ROOT / "bin" / "tileflow"):
        if candidate.exists():
            return str(candidate)
    return "tileflow"


def require_divisible(name, value, tile):
    if value <= 0 or value % tile != 0:
        raise ValueError(f"{name}={value} must be a positive multiple of {tile}")


def make_macro_yaml(path, m, n, k, output_prefix):
    require_divisible("M", m, M_TILE)
    require_divisible("N", n, N_TILE)
    require_divisible("K", k, K_TILE)
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
  MO: {m // M_TILE}
  NO: {n // N_TILE}
  KO: {k // K_TILE}
"""
    path.write_text(text)


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


def read_cycle(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == "Cycle":
                return int(float(row[1]))
    raise RuntimeError(f"Cycle not found in {csv_path}")


def run_gemm_case(tileflow_bin, raw_dir, macro_dir, stem, m, n, k):
    macro_path = macro_dir / f"{stem}.yaml"
    output_prefix = raw_dir / stem
    output_csv = output_prefix.with_suffix(".csv")
    make_macro_yaml(macro_path, m, n, k, output_prefix)
    run_tileflow(tileflow_bin, macro_path)
    return read_cycle(output_csv)
