#include "coarse.h"
#include "solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Flops per interior point per iteration:
 *   q = A p        6  (3 adds, 4*p, subtract, * 1/h^2)
 *   p.q            2
 *   x += a p       2
 *   r -= a q       2
 *   z = D^-1 r     1
 *   r.z, r.r       4
 *   p = z + b p    2                                         total 19
 * The coarse correction adds Coarse.flops_fine + Coarse.flops_solve. */
#define PCG_FLOPS 19.0

static double err_ref(const Grid *g, const double *x, const Opts *o)
{
    double m = 0.0;
    for (int j = 1; j <= g->ny; ++j)
        for (int i = 1; i <= g->nx; ++i)
            m = fmax(m, fabs(x[IDX(g, i, j)] - o->uref[IDX(g, i, j)]));
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

/* q = A p on the interior (A = -L_h, SPD); returns local p.q.
 * Ghosts of p must be current (physical-boundary ghosts are zero). */
static double matvec_dot(const Grid *g, const double *p, double *q)
{
    const size_t ld = g->ld;
    const double ih2 = g->ih2;
    double pq = 0.0;
    for (int j = 1; j <= g->ny; ++j) {
        const double *pc = p + j * ld, *ps = pc - ld, *pn = pc + ld;
        double *qc = q + j * ld;
#pragma omp simd reduction(+:pq)
        for (int i = 1; i <= g->nx; ++i) {
            double a = ih2 * (4.0 * pc[i] - (pc[i - 1] + pc[i + 1] + ps[i] + pn[i]));
            qc[i] = a;
            pq += pc[i] * a;
        }
    }
    return pq;
}

void pcg_solve(const Grid *g, double *u, const Opts *o, Stats *s)
{
    const int nx = g->nx, ny = g->ny;
    const size_t ld = g->ld;
    const double dinv = 1.0 / (4.0 * g->ih2);        /* D = diag(A) = 4/h^2 I */
    const double ts = MPI_Wtime();
    for (int d = 0; d <= MAXD; ++d) s->kres[d] = s->kerr[d] = -1;

    double *x = grid_alloc(g), *r = grid_alloc(g), *z = grid_alloc(g);
    double *p = grid_alloc(g), *q = grid_alloc(g), *ub = grid_alloc(g);
    Coarse cg;
    if (o->Nc) coarse_init(&cg, g, o->Nc);

    /* b = L_h u_b on the interior: only neighbours of the boundary are
     * non-zero since u_b vanishes inside.  Then r0 = b - A x0. */
    set_bc_only(g, ub);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            r[IDX(g, i, j)] = g->ih2 * (ub[IDX(g, i - 1, j)] + ub[IDX(g, i + 1, j)]
                                      + ub[IDX(g, i, j - 1)] + ub[IDX(g, i, j + 1)]);
            x[IDX(g, i, j)] = o->guess_rand ? guess_value(g->i0 + i, g->j0 + j) : 0.0;
        }
    if (o->guess_rand) {
        halo_exchange(g, x, &s->t);
        matvec_dot(g, x, q);
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) r[IDX(g, i, j)] -= q[IDX(g, i, j)];
    }

    /* z0 = M^-1 r0, p0 = z0 */
    double dots[2] = {0.0, 0.0};                    /* r.z, r.r */
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) z[IDX(g, i, j)] = dinv * r[IDX(g, i, j)];
    if (o->Nc) coarse_apply(&cg, g, r, z, &s->t);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            double rv = r[IDX(g, i, j)], zv = z[IDX(g, i, j)];
            p[IDX(g, i, j)] = zv;
            dots[0] += rv * zv;
            dots[1] += rv * rv;
        }
    allreduce_sum(g, dots, 2, &s->t);
    double rz = dots[0], rr0 = sqrt(dots[1]);
    if (rr0 == 0.0) rr0 = 1.0;
    double relres = sqrt(dots[1]) / rr0, relerr = -1.0;
    if (o->uref) relerr = err_ref(g, x, o);
    record(s, 0, relres, relerr);
    s->t.setup = MPI_Wtime() - ts;

    int k;
    for (k = 1; k <= o->maxit; ++k) {
        if (o->tol > 0 && relres <= o->tol) break;

        halo_exchange(g, p, &s->t);
        double pq = matvec_dot(g, p, q);
        allreduce_sum(g, &pq, 1, &s->t);
        double alpha = rz / pq;

        dots[0] = dots[1] = 0.0;
        if (!o->Nc) {
            double d0 = 0.0, d1 = 0.0;
            for (int j = 1; j <= ny; ++j) {
                double *xc = x + j * ld, *rc = r + j * ld, *zc = z + j * ld;
                const double *pc = p + j * ld, *qc = q + j * ld;
#pragma omp simd reduction(+:d0, d1)
                for (int i = 1; i <= nx; ++i) {
                    xc[i] += alpha * pc[i];
                    double rv = rc[i] - alpha * qc[i];
                    double zv = dinv * rv;
                    rc[i] = rv;
                    zc[i] = zv;
                    d0 += rv * zv;
                    d1 += rv * rv;
                }
            }
            dots[0] = d0; dots[1] = d1;
        } else {
            /* x, r, z = D^-1 r and the restriction J^T r in one sweep ... */
            double tr = MPI_Wtime();
            coarse_begin(&cg);
            for (int j = 1; j <= ny; ++j) {
                double *xc = x + j * ld, *rc = r + j * ld, *zc = z + j * ld;
                const double *pc = p + j * ld, *qc = q + j * ld;
#pragma omp simd
                for (int i = 1; i <= nx; ++i) {
                    xc[i] += alpha * pc[i];
                    rc[i] -= alpha * qc[i];
                    zc[i] = dinv * rc[i];
                }
                coarse_restrict_row(&cg, j, rc);
            }
            double tm = MPI_Wtime();
            coarse_solve(&cg, g, &s->t);
            double tp = MPI_Wtime();
            /* ... and the prolongation z += J x with the dot products */
            double d0 = 0.0, d1 = 0.0;
            for (int j = 1; j <= ny; ++j) {
                const double *rc = r + j * ld;
                double *zc = z + j * ld;
                coarse_prolong_row(&cg, j, zc);
#pragma omp simd reduction(+:d0, d1)
                for (int i = 1; i <= nx; ++i) {
                    d0 += rc[i] * zc[i];
                    d1 += rc[i] * rc[i];
                }
            }
            /* fused sweeps: cc_rp holds the whole of both sweeps */
            s->t.coarse_rp += (tm - tr) + (MPI_Wtime() - tp);
            dots[0] = d0; dots[1] = d1;
        }
        allreduce_sum(g, dots, 2, &s->t);
        double beta = dots[0] / rz;
        rz = dots[0];
        relres = sqrt(dots[1]) / rr0;

        for (int j = 1; j <= ny; ++j) {
            double *pc = p + j * ld;
            const double *zc = z + j * ld;
#pragma omp simd
            for (int i = 1; i <= nx; ++i) pc[i] = zc[i] + beta * pc[i];
        }

        if (o->uref) relerr = err_ref(g, x, o);
        record(s, k, relres, relerr);
        if (o->hist && k % o->hist == 0 && g->rank == 0)
            printf("HIST k=%d relres=%.6e relerr=%.6e\n", k, relres, relerr);
    }
    s->iters = k - 1;
    s->relres = relres;
    s->relerr = o->uref ? err_ref(g, x, o) : -1.0;
    double per_it = PCG_FLOPS * (double)(g->N - 1) * (g->N - 1);
    if (o->Nc) per_it += cg.flops_fine + cg.flops_solve;  /* redundant solve counted once */
    s->flops = per_it * s->iters;

    /* u = u0 + u_b (interior from x, ghosts from u_b) */
    memcpy(u, ub, (size_t)(nx + 2) * (ny + 2) * sizeof(double));
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) u[IDX(g, i, j)] = x[IDX(g, i, j)];

    if (o->Nc) coarse_free(&cg);
    free(x); free(r); free(z); free(p); free(q); free(ub);
}
