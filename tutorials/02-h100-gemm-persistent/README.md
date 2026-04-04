# H100 Persistent GEMM

This tutorial mirrors the H100 GEMM experiment in
`LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/H100/GEMM`
for the `BMBNBK=128x256x64` persistent kernel shape.

The setup assumes:

- H100 SXM: 132 SMs.
- 50 MB L2 cache.
- 228 KB shared memory per SM.
- FP16/BF16 data path (`word-bits: 16`).
- GEMM block tile: `BM=128`, `BN=256`, `BK=64`.

For the TileFlow spatial mesh, the SM level is modeled as a `16 x 8` grid
(`128` active slots). This matches the largest `MO x NO` sweep point in this
experiment (`16 x 8`) while staying close to the physical `132`-SM H100 SXM.

Files:

- `arch/arch.yaml`: H100-style hierarchy (`HBM -> L2 -> SharedMemory -> RegFile -> MAC`).
- `prob/prob.yaml`: GEMM problem.
- `map/map.yaml`: fixed tileflow mapping for `128x256x64`.
- `scripts/run_h100_persistent_compare.py`: sweeps the same `M/N/K` cases and writes a compare CSV.

Build:

```sh
cd /Users/jwhuang/Code/TileFlow/3rdparty/timeloop
LIBCONFIGPATH=/opt/homebrew/opt/libconfig \
YAMLCPPPATH=/opt/homebrew/opt/yaml-cpp \
NCURSESPATH=/opt/homebrew/opt/ncurses \
BOOSTDIR=/opt/homebrew/opt/boost \
scons -j4

cd /Users/jwhuang/Code/TileFlow
LIBCONFIGPATH=/opt/homebrew/opt/libconfig \
YAMLCPPPATH=/opt/homebrew/opt/yaml-cpp \
NCURSESPATH=/opt/homebrew/opt/ncurses \
BOOSTDIR=/opt/homebrew/opt/boost \
scons -j4
```

Run the sweep:

```sh
python3 tutorials/02-h100-gemm-persistent/scripts/run_h100_persistent_compare.py
```

By default the script reads:

`/Users/jwhuang/Code/LLMCompass/ref/TileDataflowAnalyticalModel/Experiment/H100/GEMM/Validation Data/Multiple_SM_BMBNBK12825664_PERSISTENT/mnk_max_block_span_512_2048_2stage.csv`

and writes:

`tutorials/02-h100-gemm-persistent/results/mnk_max_block_span_512_2048_2stage_compare.csv`
