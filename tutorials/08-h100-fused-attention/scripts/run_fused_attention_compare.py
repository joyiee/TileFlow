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

ARCH_PATH = REPO_ROOT / "tutorials/04-h100-fused-gemm/arch/arch.yaml"

FA2_ACTUAL = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/"
    "H100/FA2/results/fa2_pipeline_training_overall_max_span_compare.csv"
)
FD_ACTUAL = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/"
    "H100/FlashDecoding/results/flashdecoding_overall_max_span_compare.csv"
)
MLA_ACTUAL = Path(
    "/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/"
    "H100/FlashMLA/results/flash_mla_validation_vs_sim.csv"
)


def default_tileflow_path():
    for candidate in (REPO_ROOT / "build" / "bin" / "tileflow", REPO_ROOT / "bin" / "tileflow"):
        if candidate.exists():
            return str(candidate)
    return "tileflow"


def read_cycle(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == "Cycle":
                return int(float(row[1]))
    raise RuntimeError(f"Cycle not found in {csv_path}")


def read_cycle_from_text(text):
    for line in text.splitlines():
        if line.startswith("Cycle:"):
            return int(float(line.split(":", 1)[1].split(",", 1)[0].strip()))
        if line.startswith("Cycle,"):
            return int(float(line.split(",", 1)[1].strip()))
    raise RuntimeError("Cycle not found in TileFlow stdout")


def run_tileflow(tileflow_bin, arch_path, mapping_path, out_dir):
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = ":".join(
        [
            str(REPO_ROOT / "build" / "lib"),
            str(REPO_ROOT / "3rdparty" / "timeloop" / "lib"),
            env.get("DYLD_LIBRARY_PATH", ""),
        ]
    ).strip(":")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [tileflow_bin, str(arch_path), str(mapping_path), "-o", str(out_dir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def write_case_yaml(path, m, n, a, l, mo, ao, lo, ait, nit, title):
    # Keep the shared M/L tile outside the op-specific subtrees so each L-tile
    # can be produced by QK and immediately consumed by PV.
    qk_a_tile = ao * ait
    pv_l_tile = 4
    pv_n_tile = 2 * nit
    text = f"""problem:
  io:
    ins: Q, K, V
    outs: O
  dimensions: [M, N, A, L]
  instance:
    M: {m}
    N: {n}
    A: {a}
    L: {l}

  ops:
  - name: ProduceC
    dimensions: [M, A, L]
    data-spaces:
    - name: C
      projection:
      - [[M]]
      - [[L]]
      read-write: True
    - name: Q
      projection:
      - [[M]]
      - [[A]]
    - name: K
      projection:
      - [[A]]
      - [[L]]
    ins: Q, K
    out: C

  - name: ProduceO
    dimensions: [M, N, L]
    data-spaces:
    - name: O
      projection:
      - [[M]]
      - [[N]]
      read-write: True
    - name: C
      projection:
      - [[M]]
      - [[L]]
    - name: V
      projection:
      - [[L]]
      - [[N]]
    ins: C, V
    out: O

mapping:
  node-type: Tile
  type: Temporal
  factors: M={mo} N=1 A=1 L={lo}
  permutation: MLAN
  target: HBM

  subtree:
  - node-type: Tile
    type: Temporal
    factors: M=1 L=1
    permutation: ML
    target: L2Cache
    subtree:
    - node-type: Tile
      type: Spatial
      factors: M=16 L=2
      permutation: ML
      split: 1
      target: L2Cache
      subtree:
      - node-type: Scope
        type: Sequential
        subtree:
        - node-type: Tile
          type: Temporal
          factors: M=2 A={qk_a_tile} L=1
          permutation: AML
          target: SharedMemory
          tag: qk
          profile: False
          bypass: [C]
          subtree:
          - node-type: Tile
            type: Spatial
            factors: M=64 L=64
            permutation: ML
            split: 1
            multicast: true
            target: SharedMemory
            tag: qk
            subtree:
            - node-type: Tile
              type: Temporal
              factors: M=1 A=16 L=1
              permutation: MAL
              target: RegFile
              tag: qk
              subtree:
                - node-type: Op
                  name: ProduceC
                  binding: M:M A:A L:L

        - node-type: Tile
          type: Temporal
          factors: M=2 N={pv_n_tile} L={pv_l_tile}
          permutation: LMN
          target: SharedMemory
          tag: pv
          profile: False
          bypass: [C]
          subtree:
          - node-type: Tile
            type: Spatial
            factors: M=64 N=64
            permutation: MN
            split: 1
            multicast: true
            target: SharedMemory
            tag: pv
            subtree:
            - node-type: Tile
              type: Temporal
              factors: M=1 N=1 L=16
              permutation: MNL
              target: RegFile
              tag: pv
              subtree:
              - node-type: Op
                name: ProduceO
                binding: M:M N:N L:L

verbose: 0
"""
    path.write_text(text)


def validate_common(m, n, a, l):
    if m % 2048 != 0:
        raise ValueError(f"M={m} must be divisible by 2048")
    if l % 128 != 0:
        raise ValueError(f"L={l} must be divisible by 128")
    if n % 128 != 0:
        raise ValueError(f"N={n} must be divisible by 128")
    if a % 16 != 0:
        raise ValueError(f"A={a} must be divisible by 16")


def run_fa2(tileflow_bin, actual_csv, out_csv, yaml_dir, raw_dir):
    rows = []
    with open(actual_csv, newline="") as f:
        for row in csv.DictReader(f):
            batch = int(row["batch"])
            heads_q = int(row["heads_q"])
            heads_kv = int(row["heads_kv"])
            seq_q = int(row["seq_q"])
            seq_kv = int(row["seq_kv"])
            actual_cycles = int(float(row["actual_cycles"]))
            m = batch * heads_q * seq_q
            n = 128
            a = 128
            l = seq_kv
            validate_common(m, n, a, l)
            mo = m // 2048
            ao = 1
            lo = l // 128
            ait = a // (ao * 16)
            nit = n // 128
            stem = f"fa2_b{batch}_hq{heads_q}_hkv{heads_kv}_sq{seq_q}_sk{seq_kv}"
            yaml_path = yaml_dir / f"{stem}.yaml"
            out_dir = raw_dir / stem
            write_case_yaml(yaml_path, m, n, a, l, mo, ao, lo, ait, nit, stem)
            stdout = run_tileflow(tileflow_bin, ARCH_PATH, yaml_path, out_dir)
            sim_cycles = read_cycle_from_text(stdout)
            rows.append((batch, heads_q, heads_kv, seq_q, seq_kv, actual_cycles, sim_cycles))
            mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100.0
            print(f"FA2 {stem}: actual={actual_cycles} tileflow={sim_cycles} mape={mape:.2f}%")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "heads_q", "heads_kv", "seq_q", "seq_kv", "actual_cycles", "sim_cycles"])
        writer.writerows(rows)


def run_fd(tileflow_bin, actual_csv, out_csv, yaml_dir, raw_dir):
    rows = []
    with open(actual_csv, newline="") as f:
        for row in csv.DictReader(f):
            batch = int(row["batch"])
            heads_q = int(row["heads_q"])
            heads_kv = int(row["heads_kv"])
            seq_q = int(row["seq_q"])
            seq_kv = int(row["seq_kv"])
            actual_cycles = int(float(row["actual_cycles"]))
            m = batch * heads_q * seq_q
            n = 128
            a = 128
            l = seq_kv
            validate_common(m, n, a, l)
            mo = m // 2048
            ao = 1
            lo = l // 128
            ait = a // (ao * 16)
            nit = n // 128
            stem = f"fd_b{batch}_hq{heads_q}_hkv{heads_kv}_sq{seq_q}_sk{seq_kv}"
            yaml_path = yaml_dir / f"{stem}.yaml"
            out_dir = raw_dir / stem
            write_case_yaml(yaml_path, m, n, a, l, mo, ao, lo, ait, nit, stem)
            stdout = run_tileflow(tileflow_bin, ARCH_PATH, yaml_path, out_dir)
            sim_cycles = read_cycle_from_text(stdout)
            rows.append((batch, heads_q, heads_kv, seq_q, seq_kv, actual_cycles, sim_cycles))
            mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100.0
            print(f"FlashDecoding {stem}: actual={actual_cycles} tileflow={sim_cycles} mape={mape:.2f}%")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "heads_q", "heads_kv", "seq_q", "seq_kv", "actual_cycles", "sim_cycles"])
        writer.writerows(rows)


def run_mla(tileflow_bin, actual_csv, out_csv, yaml_dir, raw_dir):
    rows = []
    with open(actual_csv, newline="") as f:
        for row in csv.DictReader(f):
            batch = int(row["batch"])
            heads = int(row["heads"])
            seq_len = int(row["seq_len"])
            actual_cycles = int(float(row["actual_cycles"]))
            m = batch * heads
            n = 512
            a = 576
            l = seq_len
            validate_common(m, n, a, l)
            mo = m // 2048
            ao = 2
            lo = l // 128
            ait = a // (ao * 16)
            nit = n // 128
            stem = f"mla_b{batch}_h{heads}_s{seq_len}"
            yaml_path = yaml_dir / f"{stem}.yaml"
            out_dir = raw_dir / stem
            write_case_yaml(yaml_path, m, n, a, l, mo, ao, lo, ait, nit, stem)
            stdout = run_tileflow(tileflow_bin, ARCH_PATH, yaml_path, out_dir)
            sim_cycles = read_cycle_from_text(stdout)
            rows.append((batch, heads, seq_len, actual_cycles, sim_cycles))
            mape = abs(sim_cycles - actual_cycles) / actual_cycles * 100.0
            print(f"FlashMLA {stem}: actual={actual_cycles} tileflow={sim_cycles} mape={mape:.2f}%")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "heads", "seq_len", "actual_cycles", "sim_cycles"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileflow", default=default_tileflow_path())
    parser.add_argument("--mode", choices=["fa2", "flashdecoding", "flashmla"], required=True)
    parser.add_argument("--actual-csv")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    result_dir = TUTORIAL_DIR / "results"
    yaml_dir = result_dir / f"{args.mode}_yaml"
    raw_dir = result_dir / f"{args.mode}_raw"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fa2":
        actual_csv = Path(args.actual_csv) if args.actual_csv else FA2_ACTUAL
        out_csv = Path(args.out) if args.out else result_dir / "fa2_fused_compare_tileflow.csv"
        if args.limit:
            tmp = result_dir / "fa2_limit.csv"
            with open(actual_csv, newline="") as f:
                rows = list(csv.reader(f))
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(rows[0])
                w.writerows(rows[1 : 1 + args.limit])
            actual_csv = tmp
        run_fa2(args.tileflow, actual_csv, out_csv, yaml_dir, raw_dir)
    elif args.mode == "flashdecoding":
        actual_csv = Path(args.actual_csv) if args.actual_csv else FD_ACTUAL
        out_csv = Path(args.out) if args.out else result_dir / "flashdecoding_fused_compare_tileflow.csv"
        if args.limit:
            tmp = result_dir / "flashdecoding_limit.csv"
            with open(actual_csv, newline="") as f:
                rows = list(csv.reader(f))
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(rows[0])
                w.writerows(rows[1 : 1 + args.limit])
            actual_csv = tmp
        run_fd(args.tileflow, actual_csv, out_csv, yaml_dir, raw_dir)
    else:
        actual_csv = Path(args.actual_csv) if args.actual_csv else MLA_ACTUAL
        out_csv = Path(args.out) if args.out else result_dir / "flashmla_fused_compare_tileflow.csv"
        if args.limit:
            tmp = result_dir / "flashmla_limit.csv"
            with open(actual_csv, newline="") as f:
                rows = list(csv.reader(f))
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(rows[0])
                w.writerows(rows[1 : 1 + args.limit])
            actual_csv = tmp
        run_mla(args.tileflow, actual_csv, out_csv, yaml_dir, raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
