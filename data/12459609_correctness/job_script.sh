#!/bin/bash
# Correctness checks on compute nodes (2 exclusive nodes, up to 8 ranks each).
#  1. test_coarse: Galerkin stencil == J^T A J and coarse_apply == dense solve
#  2. every method gives the same solution for P = 1..16 and several process
#     grids, including uneven splits (N = 50: 49 interior points per side)
#     and the 2-node 8+8 layout used in the experiments; Jacobi must be
#     bitwise identical, CG within 1e-12.
#  3. O(h^2) convergence of the converged solution to u_e.
#SBATCH -J hw5_correct
#SBATCH -p eng-instruction
#SBATCH -A 26fa-cs598pa-eng
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --constraint=AE7713
#SBATCH -t 00:20:00
#SBATCH -o results/correctness.out
cd "$SLURM_SUBMIT_DIR"
source env.sh
set -x
hostname; date -Is
echo "native march: $(gcc -march=native -Q --help=target | awk '/^ +-march=/{print $2}')"

srun --mpi=pmix -N 1 -n 1 ./tests/test_coarse

DUMP=results/correctness_dumps
rm -rf $DUMP; mkdir -p $DUMP
N=50
# process grids "Px:Py"
GRIDS="1:1 2:1 1:2 3:1 1:3 2:2 4:1 1:4 3:2 2:3 4:2 2:4 8:1 4:4 16:1"
for m in "jacobi" "pcg" "pcg -c 10" "pcg -c 5"; do
    tag=$(echo $m | tr -d ' ')
    for gr in $GRIDS; do
        Px=${gr%:*}; Py=${gr#*:}; P=$((Px * Py))
        if [ $P -le 8 ]; then where="-N 1 -n $P"; else where="-N 2 -n $P --ntasks-per-node=8"; fi
        srun --mpi=pmix $where ./poisson -m $m -N $N -x $Px -y $Py -t 1e-10 -R \
             -o $DUMP/u_${tag}_${Px}x${Py}_P${P}.bin | grep -E 'RESULT'
    done
done
python3 tests/compare.py $DUMP

# O(h^2): converged CG (tol 1e-13) against u_e
for N in 16 32 64 128 256 512 1024; do
    srun --mpi=pmix -N 1 -n 1 ./poisson -m pcg -N $N -t 1e-13 | grep RESULT
done
date -Is
