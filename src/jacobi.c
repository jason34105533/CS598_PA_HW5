#include "solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JACOBI_FLOPS 5.0   /* 3 adds + 1 mul (stencil) + 1 sub (update) */

double guess_value(int I, int J)
{
    unsigned long long x = (unsigned long long)I * 0x9E3779B97F4A7C15ull
                         ^ (unsigned long long)J * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33;
    return (double)(x >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static double err_ref(const Grid *g, const double *u, const Opts *o)
{
    double m = 0.0;
    for (int j = 1; j <= g->ny; ++j)
        for (int i = 1; i <= g->nx; ++i)
            m = fmax(m, fabs(u[IDX(g, i, j)] - o->uref[IDX(g, i, j)]));
    double y;
    MPI_Allreduce(&m, &y, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    return y / o->refnorm;
}

static void record(Stats *s, int k, double relres, double relerr)
{
    for (int d = 1; d <= MAXD; ++d) {
        double tol = pow(10.0, -d);
        if (s->kres[d] < 0 && relres <= tol) s->kres[d] = k;
        if (relerr >= 0 && s->kerr[d] < 0 && relerr <= tol) s->kerr[d] = k;
    }
}

/*
 * One sweep computes u_new = (sum of 4 neighbours)/4 and
 * max |u_new - u| = (h^2/4) ||r(u)||_inf, i.e. the residual of the iterate
 * *before* the sweep.  So after sweep k we know ||r_{k-1}||.  The loop stops
 * at the first sweep whose input iterate u_{k-1} satisfies the tolerance and
 * returns iters = k-1 (the extra sweep's output, which is no worse, is what
 * is returned in u).  With tol <= 0 exactly maxit sweeps are done.
 */
void jacobi_solve(const Grid *g, double *u, const Opts *o, Stats *s)
{
    const int nx = g->nx, ny = g->ny;
    const size_t ld = g->ld;
    double *v = grid_alloc(g);
    for (int d = 0; d <= MAXD; ++d) s->kres[d] = s->kerr[d] = -1;

    set_boundary(g, u);
    set_boundary(g, v);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            u[IDX(g, i, j)] = o->guess_rand ? guess_value(g->i0 + i, g->j0 + j) : 0.0;

    double r0 = 0.0, relres = 1.0, relerr = -1.0;
    int k;
    for (k = 1; k <= o->maxit; ++k) {
        halo_exchange(g, u, &s->t);
        double dmax = 0.0;
        for (int j = 1; j <= ny; ++j) {
            const double *uc = u + j * ld, *us = uc - ld, *un = uc + ld;
            double *vc = v + j * ld;
#pragma omp simd reduction(max:dmax)
            for (int i = 1; i <= nx; ++i) {
                double w = 0.25 * (uc[i - 1] + uc[i + 1] + us[i] + un[i]);
                dmax = fmax(dmax, fabs(w - uc[i]));
                vc[i] = w;
            }
        }
        dmax = allreduce_max(g, dmax, &s->t);       /* ||r_{k-1}|| * h^2/4 */
        if (k == 1) r0 = dmax > 0 ? dmax : 1.0;
        relres = dmax / r0;

        if (o->uref) {                              /* error of u_{k-1} */
            relerr = err_ref(g, u, o);
        }
        record(s, k - 1, relres, relerr);
        if (o->hist && (k - 1) % o->hist == 0 && g->rank == 0)
            printf("HIST k=%d relres=%.6e relerr=%.6e\n", k - 1, relres, relerr);

        double *tmp = u; u = v; v = tmp;            /* u = u_k */
        if (o->tol > 0 && relres <= o->tol) break;
    }
    int sweeps = k > o->maxit ? o->maxit : k;
    s->iters = (o->tol > 0 && relres <= o->tol) ? sweeps - 1 : sweeps;
    s->relres = relres;
    s->relerr = o->uref ? err_ref(g, u, o) : -1.0;
    s->flops = JACOBI_FLOPS * (double)(g->N - 1) * (g->N - 1) * sweeps;

    /* After an odd number of swaps the result lives in the scratch array. */
    if (sweeps % 2) {
        memcpy(v, u, (size_t)(nx + 2) * (ny + 2) * sizeof(double));
        free(u);
    } else {
        free(v);
    }
}
