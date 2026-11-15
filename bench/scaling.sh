#!/bin/bash
# Parallel efficiency vs n/P for jacobi, pcg and pcg+coarse (Nc = 32), fixed
# iteration count (bench/common.sh: fixed_iters), nrep = 3, fastest reported.
#   P=1                 1 rank, core 0 (baseline T_1)
#   P=8   1 node        ccd layout (8 CCDs)
#   P=16  1 node        ccd layout (16 CCDs)
#   P=16  2 nodes x 8   ccd layout on each node   <- the assignment's case
#   P=16  2 nodes x 8   ablations: -A no halo exchange, -B no scalar all-reduce
#SBATCH -J hw5_scaling
#SBATCH -p eng-instruction
#SBATCH -A 26fa-cs598pa-eng
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --constraint=AE7713
#SBATCH -t 03:00:00
#SBATCH -o results/scaling.out
cd "$SLURM_SUBMIT_DIR"
source env.sh
source bench/common.sh
hostname; date -Is
echo "native march: $(gcc -march=native -Q --help=target | awk '/^ +-march=/{print $2}')"
env | grep -E '^UCX_|^OMPI_MCA' | sort

run() {  # nodes ranks_per_node args...
    local nn=$1 ppn=$2; shift 2
    echo "== nodes=$nn ppn=$ppn $*"
    srun --mpi=pmix -N $nn --ntasks-per-node=$ppn -n $((nn * ppn)) \
         --cpu-bind=map_cpu:$(cpumap $ppn) ./poisson "$@" -t 0 -r 3 | grep -E 'REP|RESULT'
}

for N in 64 128 256 512 1024 2048 4096 8192 16384; do
    K=$(fixed_iters $N)
    for m in "jacobi" "pcg" "pcg -c 32"; do
        run 1 1  -m $m -N $N -k $K
        run 1 8  -m $m -N $N -k $K
        run 1 16 -m $m -N $N -k $K
        run 2 8  -m $m -N $N -k $K
        run 2 8  -m $m -N $N -k $K -A
        run 2 8  -m $m -N $N -k $K -B
    done
done
date -Is
