#!/bin/bash
# Problems (1) and (3): iteration counts to reach d digits, serial (P=1).
# Only iteration counts are recorded (no timing), so independent runs share
# the node, one core each.  -R computes the exact discrete solution u* so the
# true algebraic error ||u_k - u*||/||u*|| is tracked next to the residual;
# CROSS lines give the first k with relres <= 1e-d and relerr <= 1e-d.
#SBATCH -J hw5_q13
#SBATCH -p eng-instruction
#SBATCH -A 26fa-cs598pa-eng
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --exclusive
#SBATCH --constraint=AE7713
#SBATCH -t 02:00:00
#SBATCH -o results/q13_convergence.out
cd "$SLURM_SUBMIT_DIR"
source env.sh
BIN=$PWD/results/bin/poisson.$SLURM_JOB_ID   # private copy of the binary
mkdir -p results/bin && cp ./poisson $BIN && sha256sum $BIN
source bench/common.sh
hostname; date -Is
OUT=results/q13
rm -rf $OUT; mkdir -p $OUT

one() {  # tag args...   -> results/q13/<tag>.out
    local tag=$1; shift
    srun --mpi=pmix -n 1 --exact --cpu-bind=cores $BIN "$@" > $OUT/$tag.out 2>&1 &
}

# Jacobi, zero and random initial guess; residual tol 1e-13 so the error
# (which lags the residual) reaches ~10 digits.
for N in 16 32 64 128 256 512; do
    one jacobi_zero_N$N -m jacobi -N $N -t 1e-13 -R
done
for N in 16 32 64 128 256; do
    one jacobi_rand_N$N -m jacobi -N $N -t 1e-13 -R -g
done
one jacobi_hist_N64 -m jacobi -N 64 -t 1e-13 -R -H 50

# JPCG, zero and random initial guess.
for N in 16 32 64 128 256 512 1024 2048; do
    one pcg_zero_N$N -m pcg -N $N -t 1e-14 -k 100000 -R
    one pcg_rand_N$N -m pcg -N $N -t 1e-14 -k 100000 -R -g
done
one pcg_hist_N256 -m pcg -N 256 -t 1e-14 -k 100000 -R -H 1

# JPCG + coarse correction (problem 4): fixed ratio m = H/h and fixed Nc.
for N in 32 64 128 256 512 1024 2048; do
    for m in 4 8 16; do
        [ $((N / m)) -ge 2 ] && one pcgc_m${m}_N$N -m pcg -N $N -c $((N / m)) -t 1e-14 -k 100000 -R
    done
    for Nc in 8 16; do
        [ $Nc -lt $N ] && one pcgc_Nc${Nc}_N$N -m pcg -N $N -c $Nc -t 1e-14 -k 100000 -R
    done
done
wait
cat $OUT/*.out
date -Is
