# Archived experiment data

Read-only raw data, one directory per Slurm job: `data/<jobid>_<name>/`,
created by `bench/archive_job.sh`; never edit these files. Each directory
holds the raw job output, `job_script.sh` (the exact batch script Slurm ran,
from `sacct -B`), `sacct.txt` (nodes, start/end, state) and
`provenance.txt` (sha256 of sources, binaries and scripts at archive time).
`MANIFEST.sha256` lists every archived file:

    sha256sum -c data/MANIFEST.sha256

All jobs: `eng-instruction` partition, exclusive AE7713 nodes
(2x AMD EPYC 7713, 128 cores), gcc 13.3.0 + Open MPI 5.0.1, `-O3 -march=znver3`.

| Job | Name | Nodes | What | Outcome |
|---|---|---|---|---|
| 12459779 | correctness | ccc[0391-0392] | `test_coarse`; jacobi / pcg / pcg+coarse (Nc=10, 5) at N=50 on 15 process grids (P=1..16, uneven splits, 2-node 8+8); O(h^2) check N=16..1024 | all PASS: Jacobi bitwise identical on every grid, CG max diff 3.6e-15; error ratio 3.98-4.00 per halving of h |
| 12460027 | q13_convergence | ccc0391 | serial iteration counts vs d: Jacobi N=16..512, JPCG N=16..2048 (zero and random guess), JPCG+coarse (H/h = 4, 8, 16 and Nc = 8, 16), all with `-R` | 64 runs completed |
| 12460028 | scaling | ccc[0392-0393] | jacobi / pcg / pcg+coarse (Nc=32), N=64..16384, fixed iterations, nrep=3: P=1, P=8 and P=16 on 1 node, P=16 on 2 nodes x 8, plus `-A` / `-B` ablations | 162 runs completed |
| 12460029 | coarse | ccc[0391-0392] | JPCG with/without coarse correction, Nc = 8..256, N = 1024/2048/4096: (a) time to solution at tol 1e-8, P=16 (2 nodes x 8) and P=1 (N <= 2048); (b) 200 fixed iterations, P=16, cost breakdown | 56 runs completed |
