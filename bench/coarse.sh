#!/bin/bash
# Problem (4): JPCG with and without the coarse correction, sweep Nc.
#   a) time to solution, tol 1e-8, P=16 on 2 nodes x 8 and P=1
#   b) per-iteration cost breakdown at fixed iteration count (P=16, 2 nodes)
# nrep = 3 (2 for the slow serial solves), fastest reported.
#SBATCH -J hw5_coarse
#SBATCH -p eng-instruction
#SBATCH -A 26fa-cs598pa-eng
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --constraint=AE7713
#SBATCH -t 02:00:00
#SBATCH -o results/coarse.out
cd "$SLURM_SUBMIT_DIR"
source env.sh
source bench/common.sh
hostname; date -Is

run() {  # nodes ranks_per_node args...
    local nn=$1 ppn=$2; shift 2
    echo "== nodes=$nn ppn=$ppn $*"
    srun --mpi=pmix -N $nn --ntasks-per-node=$ppn -n $((nn * ppn)) \
         --cpu-bind=map_cpu:$(cpumap $ppn) ./poisson "$@" | grep -E 'REP|RESULT'
}

echo "#### a) time to solution, tol 1e-8"
for N in 1024 2048 4096; do
    for c in 0 8 16 32 64 128 256; do
        run 2 8 -m pcg -N $N -c $c -t 1e-8 -r 3
        [ $N -le 2048 ] && run 1 1 -m pcg -N $N -c $c -t 1e-8 -r 2
    done
done

echo "#### b) fixed 200 iterations, cost breakdown"
for N in 1024 2048 4096; do
    for c in 0 8 16 32 64 128 256; do
        run 2 8 -m pcg -N $N -c $c -t 0 -k 200 -r 3
    done
done
date -Is
