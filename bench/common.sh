# Shared helpers for benchmark jobs (source after env.sh).
# EPYC 7713 node: 16 CCDs x 8 cores, one 32 MB L3 per CCD; cores 0-63 on
# socket 0, 64-127 on socket 1.  "ccd" layout (as in HW3): rank r -> one core
# per CCD first, alternating sockets, so 8 ranks per node get 8 different
# CCDs, 4 per socket, and 16 ranks get all 16 CCDs.
cpumap() {  # nranks_per_node -> comma-separated core list
    awk -v N=$1 'BEGIN{
        for (r=0; r<N; ++r) {
            c = ((r%2)*8 + int(r/2)%8)*8 + int(r/16);
            printf "%s%d", (r ? "," : ""), c;
        }}'
}
export OMP_NUM_THREADS=1
# Fixed iteration count for timing runs: about 2^29 point-iterations, >= 20.
fixed_iters() {  # N
    awk -v N=$1 'BEGIN{ n=(N-1)*(N-1); k=int(2^29/n); if (k<20) k=20; print k }'
}
