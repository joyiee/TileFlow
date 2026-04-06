# H100 Fused GEMM + SiLU

This tutorial models `GEMM + SiLU` on H100 as a fused TileFlow case.

## What Is Modeled

The workload is expressed as two ops in one TileFlow problem:

1. `GEMM`: `A[M,K] x B[K,N] -> C[M,N]`
2. `SiLU`: `C[M,N] -> D[M,N]`

The mapping then places the two ops under one `Scope: Sequential`, so TileFlow evaluates them as a fused epilogue-style dataflow at tile granularity instead of two fully separate top-level kernels.

In pseudocode, the intent is:

```python
for m_tile, n_tile:
    C_tile = gemm(A_tile, B_tile)
    D_tile = silu_proxy(C_tile)
```

## Important Modeling Note

This tutorial does **not** implement the full mathematical SiLU chain:

```python
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Instead, the `SiLU` op here is a **generic pointwise epilogue proxy**:

- it reads `C[M,N]`
- it writes `D[M,N]`
- it preserves the post-GEMM read/write relationship
- it does **not** explicitly model `exp`, `add`, `reciprocal/div`, or the exact Hopper epilogue instruction sequence

So this experiment should be interpreted as:

`GEMM + pointwise SiLU epilogue proxy`

not as a cycle-accurate implementation of the true fused Hopper kernel.

## Files

- `arch/arch.yaml`: H100 architecture model
- `prob/prob.yaml`: two-op fused workload (`GEMM`, `SiLU`)
- `map/map.yaml`: fused mapping with `Scope: Sequential`
- `scripts/run_h100_fused_gemm_compare.py`: batch runner over `(M,N,K)` workloads
- `results/mnk_max_block_span_512_2048_fused_silu_compare.csv`: actual vs TileFlow cycles

## Run

```bash
python3 -u tutorials/04-h100-fused-gemm/scripts/run_h100_fused_gemm_compare.py
```
