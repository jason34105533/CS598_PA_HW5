# CS598 HPC — HW5: Domain Decomposition I

Iterative solvers for the 2D Laplace problem on Ω = [0,1]² with Dirichlet data
u_e(x,y) = sin(2x)e^(−2y), second-order finite differences, h = 1/N, on a
Px × Py MPI process grid. The assignment is in [`docs/hw5.pdf`](docs/hw5.pdf).

| Problem | Method | Where |
|---|---|---|
| (1) | serial Jacobi, k_max ≈ dκ | `src/jacobi.c` (P = 1) |
| (2) | distributed Jacobi: halo exchange + global max all-reduce | `src/jacobi.c`, `src/grid.c` |
| (3) | Jacobi-preconditioned CG, A u₀ = b with b = −A_grid u_b, k_max ≈ d√κ | `src/pcg.c` |
| (4) | JPCG + two-level additive coarse correction M⁻¹ = D⁻¹ + J A_c⁻¹ Jᵀ | `src/coarse.c` |

## Layout

```
src/        solver (one binary, `poisson`)
  grid.[ch]   2D decomposition, halo exchange, all-reduces, boundary data, timers
  jacobi.c    Jacobi iteration
  pcg.c       JPCG (optionally with the coarse correction)
  coarse.[ch] bilinear J, Galerkin A_c = JᵀAJ, banded Cholesky, restriction/prolongation
  refsol.c    exact discrete solution u* (dense sine transform), for measuring the algebraic error
  main.c      command line, repetitions, timing, output
tests/      correctness: coarse-grid unit test, cross-process-grid comparison, Slurm job
bench/      Slurm experiment jobs and helpers
data/       archived raw output of every job used (read-only, see data/README.md)
results/    working copies of job output (not in git)
docs/       assignment
```

## Build

On the Campus Cluster (compile on the login node, run only through Slurm):

```
source env.sh      # gcc 13.3.0, Open MPI 5.0.1
make               # -> ./poisson
make tests/test_coarse
```

The compute nodes are EPYC 7713 (Zen 3) and the login node is Zen 4, so the
Makefile targets `-march=znver3` instead of `-march=native`.

## Discretisation and solvers

- Unknowns are the (N−1)² interior points; each rank owns an nx × ny block
  plus one ghost layer (x unit-stride). Ghosts on the physical boundary hold
  boundary data; the others are filled by a nonblocking 4-neighbour exchange.
- **Jacobi** sweeps u ← (sum of 4 neighbours)/4 and monitors
  ‖r‖∞/‖r₀‖∞ with one `MPI_Allreduce(MAX)` per iteration.
- **JPCG** solves A u₀ = b with A = −L_h (SPD) and b = L_h u_b, where u_b is
  zero inside and u_e on the boundary; the matrix-vector product is applied as
  a grid function with zero boundary ghosts, and the solution returned is
  u₀ + u_b. Two all-reduces per iteration (pᵀAp, and rᵀz with rᵀr fused).
  Since D = diag(A) = (4/h²)I is a constant here, JPCG takes exactly the same
  iterations as unpreconditioned CG.
- **Coarse correction** (`-c Nc`): coarse grid H = 1/Nc (N divisible by Nc),
  J = bilinear interpolation, A_c = JᵀAJ (a constant 9-point stencil, checked
  against an explicit JᵀAJ in `tests/test_coarse.c`). A_c is factored once by
  banded Cholesky on every rank; each application restricts locally, sums the
  (Nc+1)² coarse vector with one `MPI_Allreduce(SUM)`, solves redundantly and
  prolongates locally.
- `-R` computes the exact discrete solution u* by a dense O(N³) sine
  transform, so the true algebraic error ‖u_k − u*‖∞/‖u*‖∞ can be tracked
  alongside the residual.

## Running

```
poisson -m jacobi|pcg -N N [options]
  -t TOL      relative residual tolerance (0: run exactly MAXIT iterations)
  -k MAXIT    iteration cap / fixed iteration count
  -c NC       coarse grid with NC intervals (pcg only)
  -x PX -y PY process grid (default: MPI_Dims_create)
  -r NREP     repetitions, the fastest is reported
  -R          track the error against the exact discrete solution
  -H S        print residual/error history every S iterations
  -g          pseudo-random initial guess (default zero)
  -A, -B      ablation: skip the halo exchange / the scalar all-reduces
              (the numbers are then wrong; used only to time communication)
  -o FILE     write the (N+1)² solution as raw doubles
```

Output is one `key=value` line per record, from rank 0:

| Line | Content |
|---|---|
| `CONF` | method, N, P, Px×Py, number of nodes, κ = cot²(π/2N), ... |
| `REF` | ‖u*‖∞ and ‖u* − u_e‖∞ (with `-R`) |
| `HIST` | k, relative residual, relative error (with `-H`) |
| `REP` | one per repetition: loop time (max over ranks), setup, per-rank mean halo / all-reduce / coarse times |
| `CROSS` | first k with relres ≤ 10⁻ᵈ and relerr ≤ 10⁻ᵈ, d = 1..16 (with `-R`) |
| `RESULT` | fastest repetition: iterations, time, t_iter, GFLOP/s, timers, final residual/error, ‖u − u_e‖∞ |

`time` covers only the iteration loop; solver setup (coarse factorisation,
initial residual) is reported separately as `setup`. Flop counts per point
and iteration: Jacobi 5, JPCG 19, coarse correction +8 plus 4·n_c·b_w for the
banded solve (counted once, not per rank).

## Experiments

All runs use exclusive AE7713 nodes of `eng-instruction` (2× EPYC 7713,
128 cores, 16 CCDs). Each job runs from a private copy of the binary,
`results/bin/poisson.<jobid>`. Ranks are pinned one per CCD, alternating
sockets (`bench/common.sh`), so 8 ranks per node sit on 8 different CCDs.

| Script | Purpose |
|---|---|
| `tests/correctness.sh` | `test_coarse`; identical solutions on 15 process grids (P = 1..16, uneven splits, 2 nodes × 8); O(h²) convergence |
| `bench/q13_convergence.sh` | iterations to d digits vs κ for Jacobi, JPCG and JPCG + coarse (serial, `-R`) |
| `bench/scaling.sh` | parallel efficiency vs n/P at a fixed iteration count: P = 1, 8, 16 on one node and P = 16 on 2 nodes × 8, plus halo / all-reduce ablations |
| `bench/coarse.sh` | coarse grid size sweep: time to solution (tol 10⁻⁸) and per-iteration cost breakdown |

Submit with `sbatch <script>` from the repository root. After a job finishes,
`bench/archive_job.sh <jobid> <name> <files>` copies its output to
`data/<jobid>_<name>/` with the batch script, `sacct` record and source
hashes.
