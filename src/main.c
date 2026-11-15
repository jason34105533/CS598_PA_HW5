/*
 * poisson: Jacobi / JPCG / JPCG + coarse-grid correction for the 2D Laplace
 * problem of CS598 HW5 on a Px x Py MPI process grid.
 *
 *   poisson -m jacobi|pcg -N N [options]
 *     -m METHOD   jacobi | pcg                       (default pcg)
 *     -N N        h = 1/N, (N-1)^2 unknowns          (default 128)
 *     -t TOL      relative residual tolerance; 0 = fixed iteration count
 *     -k MAXIT    iteration cap / fixed count        (default 10^7)
 *     -c NC       coarse grid with NC intervals (pcg only, 0 = off)
 *     -x PX -y PY process grid (default MPI_Dims_create)
 *     -r NREP     repetitions; the fastest is reported (default 1)
 *     -R          compute the exact discrete solution and track the error
 *     -H S        print the residual/error history every S iterations
 *     -g          random initial guess instead of zero
 *     -A          ablation: skip the halo exchange (results are wrong)
 *     -B          ablation: skip the scalar all-reduces (results are wrong)
 *     -o FILE     write the final solution (raw doubles, (N+1)^2) on rank 0
 *
 * Output (rank 0, key=value): CONF, optional HIST/CROSS, one REP per
 * repetition and a final RESULT line.
 */
#include "solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

static void usage(void)
{
    fprintf(stderr, "usage: poisson -m jacobi|pcg -N N [-t tol] [-k maxit] [-c Nc] "
                    "[-x Px -y Py] [-r nrep] [-R] [-H s] [-g] [-A] [-B] [-o file]\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    /* Flush denormals so the ablation runs (which may produce garbage
     * values) cannot hit slow paths; normal runs never see denormals. */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    char method[16] = "pcg", *outfile = NULL;
    int N = 128, Px = 0, Py = 0, nrep = 1, want_ref = 0, nohalo = 0, noreduce = 0;
    Opts o = {.maxit = 10000000, .tol = 1e-8};
    int opt;
    while ((opt = getopt(argc, argv, "m:N:t:k:c:x:y:r:RH:gABo:")) != -1) {
        switch (opt) {
        case 'm': snprintf(method, sizeof method, "%s", optarg); break;
        case 'N': N = atoi(optarg); break;
        case 't': o.tol = atof(optarg); break;
        case 'k': o.maxit = atoi(optarg); break;
        case 'c': o.Nc = atoi(optarg); break;
        case 'x': Px = atoi(optarg); break;
        case 'y': Py = atoi(optarg); break;
        case 'r': nrep = atoi(optarg); break;
        case 'R': want_ref = 1; break;
        case 'H': o.hist = atoi(optarg); break;
        case 'g': o.guess_rand = 1; break;
        case 'A': nohalo = 1; break;
        case 'B': noreduce = 1; break;
        case 'o': outfile = optarg; break;
        default: usage();
        }
    }
    int is_jac = !strcmp(method, "jacobi");
    if (!is_jac && strcmp(method, "pcg")) usage();
    if (is_jac && o.Nc) usage();

    Grid g;
    grid_init(&g, N, Px, Py, MPI_COMM_WORLD);
    char host[64];
    gethostname(host, sizeof host);
    host[sizeof host - 1] = 0;
    int nodes = 0;
    {   /* count distinct hosts */
        char *all = g.rank == 0 ? malloc((size_t)g.P * 64) : NULL;
        MPI_Gather(host, 64, MPI_CHAR, all, 64, MPI_CHAR, 0, g.comm);
        if (g.rank == 0) {
            for (int r = 0; r < g.P; ++r) {
                int seen = 0;
                for (int q = 0; q < r; ++q) seen |= !strcmp(all + 64 * r, all + 64 * q);
                nodes += !seen;
            }
            free(all);
        }
    }

    const double kappa = 1.0 / pow(tan(M_PI / (2.0 * N)), 2);
    const double n = (double)(N - 1) * (N - 1);
    if (g.rank == 0) {
        printf("CONF method=%s N=%d P=%d Px=%d Py=%d nodes=%d Nc=%d tol=%g maxit=%d "
               "nrep=%d guess=%s halo=%d reduce=%d kappa=%.6e\n",
               is_jac ? "jacobi" : (o.Nc ? "pcgc" : "pcg"), N, g.P, g.Px, g.Py, nodes,
               o.Nc, o.tol, o.maxit, nrep, o.guess_rand ? "rand" : "zero",
               !nohalo, !noreduce, kappa);
        fflush(stdout);
    }

    double *uref = NULL;
    if (want_ref) {
        uref = ref_solution(&g);
        Timers dummy = {0};
        o.uref = uref;
        o.refnorm = norm_inf(&g, uref, &dummy);
        double eref = err_vs_exact(&g, uref);
        if (g.rank == 0) printf("REF refnorm=%.6e disc_err=%.6e\n", o.refnorm, eref);
    }

    double *u = grid_alloc(&g);
    Stats best = {0};
    double best_t = 1e300;
    for (int rep = 0; rep < nrep; ++rep) {
        Stats s = {0};
        memset(u, 0, (size_t)(g.nx + 2) * (g.ny + 2) * sizeof(double));
        g.halo_on = !nohalo;
        g.reduce_on = !noreduce;
        MPI_Barrier(g.comm);
        double t0 = MPI_Wtime();
        if (is_jac) jacobi_solve(&g, u, &o, &s);
        else        pcg_solve(&g, u, &o, &s);
        s.t.total = MPI_Wtime() - t0;
        g.halo_on = g.reduce_on = 1;

        /* reduce timers: max and mean over ranks */
        double loc[6] = {s.t.total, s.t.halo, s.t.reduce, s.t.coarse_comm,
                         s.t.coarse_solve, s.t.coarse_rp};
        double mx[6], sm[6];
        MPI_Reduce(loc, mx, 6, MPI_DOUBLE, MPI_MAX, 0, g.comm);
        MPI_Reduce(loc, sm, 6, MPI_DOUBLE, MPI_SUM, 0, g.comm);
        MPI_Bcast(mx, 6, MPI_DOUBLE, 0, g.comm);
        if (g.rank == 0) {
            for (int i = 0; i < 6; ++i) sm[i] /= g.P;
            printf("REP rep=%d time=%.6e halo=%.6e reduce=%.6e cc_comm=%.6e cc_solve=%.6e "
                   "cc_rp=%.6e halo_max=%.6e reduce_max=%.6e iters=%d gflops=%.4f\n",
                   rep, mx[0], sm[1], sm[2], sm[3], sm[4], sm[5], mx[1], mx[2], s.iters,
                   s.flops / mx[0] * 1e-9);
            fflush(stdout);
        }
        s.t.total = mx[0];
        s.t.halo = sm[1]; s.t.reduce = sm[2];
        s.t.coarse_comm = sm[3]; s.t.coarse_solve = sm[4]; s.t.coarse_rp = sm[5];
        if (mx[0] < best_t) { best_t = mx[0]; best = s; }
    }

    double eex = err_vs_exact(&g, u);
    if (g.rank == 0) {
        if (want_ref)
            for (int d = 1; d <= MAXD; ++d)
                printf("CROSS d=%d kres=%d kerr=%d\n", d, best.kres[d], best.kerr[d]);
        printf("RESULT method=%s N=%d n=%.0f P=%d Px=%d Py=%d nodes=%d Nc=%d n_per_P=%.1f "
               "iters=%d time=%.6e t_iter=%.6e gflops=%.4f halo=%.6e reduce=%.6e "
               "cc_comm=%.6e cc_solve=%.6e cc_rp=%.6e relres=%.3e relerr=%.3e "
               "err_exact=%.6e kappa=%.6e halo_on=%d reduce_on=%d nrep=%d\n",
               is_jac ? "jacobi" : (o.Nc ? "pcgc" : "pcg"), N, n, g.P, g.Px, g.Py, nodes,
               o.Nc, n / g.P, best.iters, best.t.total,
               best.iters ? best.t.total / best.iters : 0.0, best.flops / best.t.total * 1e-9,
               best.t.halo, best.t.reduce, best.t.coarse_comm, best.t.coarse_solve,
               best.t.coarse_rp, best.relres, best.relerr, eex, kappa, !nohalo, !noreduce, nrep);
        fflush(stdout);
    }

    if (outfile) {
        double *gl = gather_interior(&g, u);
        if (g.rank == 0) {
            FILE *f = fopen(outfile, "wb");
            int Np = N + 1;
            double *full = malloc((size_t)Np * Np * sizeof(double));
            for (int J = 0; J <= N; ++J)
                for (int I = 0; I <= N; ++I)
                    full[I + (size_t)J * Np] = (I == 0 || J == 0 || I == N || J == N)
                        ? ue((double)I / N, (double)J / N)
                        : gl[(I - 1) + (size_t)(J - 1) * (N - 1)];
            fwrite(full, sizeof(double), (size_t)Np * Np, f);
            fclose(f);
            free(full);
            free(gl);
        }
    }

    free(u);
    free(uref);
    grid_free(&g);
    MPI_Finalize();
    return 0;
}
