#include "coarse.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double hat(double t) { t = fabs(t); return t < 1.0 ? 1.0 - t : 0.0; }

/* Fine-grid hat function of the coarse node at fine offset (ox, oy). */
static double phi(int I, int J, int ox, int oy, int m)
{
    return hat((double)(I - ox) / m) * hat((double)(J - oy) / m);
}

/* Ac(0, d) = phi_0^T A phi_d, A = (1/h^2)(4 - shifts).  Every interior coarse
 * hat vanishes on the boundary, so this is the same for all coarse nodes. */
static void galerkin_stencil(double st[3][3], int m, double ih2)
{
    for (int dJ = -1; dJ <= 1; ++dJ)
        for (int dI = -1; dI <= 1; ++dI) {
            int ox = dI * m, oy = dJ * m;
            double s = 0.0;
            for (int J = -m + 1; J <= m - 1; ++J)
                for (int I = -m + 1; I <= m - 1; ++I) {
                    double Ap = 4.0 * phi(I, J, ox, oy, m)
                              - phi(I - 1, J, ox, oy, m) - phi(I + 1, J, ox, oy, m)
                              - phi(I, J - 1, ox, oy, m) - phi(I, J + 1, ox, oy, m);
                    s += phi(I, J, 0, 0, m) * Ap;
                }
            st[dJ + 1][dI + 1] = ih2 * s;
        }
}

/* Row-banded Cholesky: L[i*(bw+1) + (k - i + bw)] = L(i, k), k in [i-bw, i]. */
#define LB(c, i, k) ((c)->L[(size_t)(i) * ((c)->bw + 1) + (size_t)((k) - (i) + (c)->bw)])

static void factor(Coarse *c)
{
    const int n = c->nc, bw = c->bw, n1 = c->n1;
    c->L = calloc((size_t)n * (bw + 1), sizeof(double));
    /* assemble the lower band of Ac */
    for (int i = 0; i < n; ++i) {
        int I = i % n1, J = i / n1;
        for (int dJ = -1; dJ <= 0; ++dJ)
            for (int dI = -1; dI <= 1; ++dI) {
                if (dJ == 0 && dI > 0) continue;
                int I2 = I + dI, J2 = J + dJ;
                if (I2 < 0 || I2 >= n1 || J2 < 0) continue;
                LB(c, i, I2 + J2 * n1) = c->stencil[dJ + 1][dI + 1];
            }
    }
    /* in-place Cholesky within the band */
    for (int i = 0; i < n; ++i) {
        int k0 = i - bw > 0 ? i - bw : 0;
        for (int j = k0; j <= i; ++j) {
            double s = LB(c, i, j);
            int kk = (j - bw > k0) ? j - bw : k0;
            for (int k = kk; k < j; ++k) s -= LB(c, i, k) * LB(c, j, k);
            if (j < i) {
                LB(c, i, j) = s / LB(c, j, j);
            } else {
                if (s <= 0) { fprintf(stderr, "coarse Cholesky: not SPD\n"); abort(); }
                LB(c, i, i) = sqrt(s);
            }
        }
    }
}

/* Solve Ac x = b in place (b: nc entries). */
static void band_solve(const Coarse *c, double *b)
{
    const int n = c->nc, bw = c->bw;
    for (int i = 0; i < n; ++i) {
        int k0 = i - bw > 0 ? i - bw : 0;
        double s = b[i];
        const double *Li = &c->L[(size_t)i * (bw + 1) + (k0 - i + bw)];
        for (int k = k0; k < i; ++k) s -= Li[k - k0] * b[k];
        b[i] = s / LB(c, i, i);
    }
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= LB(c, i, i);
        int k0 = i - bw > 0 ? i - bw : 0;
        const double *Li = &c->L[(size_t)i * (bw + 1) + (k0 - i + bw)];
        double xi = b[i];
        for (int k = k0; k < i; ++k) b[k] -= Li[k - k0] * xi;
    }
}

void coarse_init(Coarse *c, const Grid *g, int Nc)
{
    memset(c, 0, sizeof *c);
    if (Nc < 2 || g->N % Nc) {
        if (g->rank == 0) fprintf(stderr, "coarse: need Nc >= 2 dividing N\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    c->Nc = Nc;
    c->m = g->N / Nc;
    c->n1 = Nc - 1;
    c->nc = c->n1 * c->n1;
    c->bw = c->n1 + 1;
    galerkin_stencil(c->stencil, c->m, g->ih2);
    factor(c);

    const int nx = g->nx, ny = g->ny, m = c->m;
    c->rc  = malloc((size_t)(Nc + 1) * (Nc + 1) * sizeof(double));
    c->row = malloc((size_t)(Nc + 1) * sizeof(double));
    c->cy  = malloc((size_t)(ny + 2) * sizeof(int));
    c->wy  = malloc((size_t)(ny + 2) * sizeof(double));
    c->w0  = malloc((size_t)(nx + 2) * sizeof(double));
    c->w1  = malloc((size_t)(nx + 2) * sizeof(double));
    c->seg_c  = malloc((size_t)(Nc + 1) * sizeof(int));
    c->seg_lo = malloc((size_t)(Nc + 1) * sizeof(int));
    c->seg_hi = malloc((size_t)(Nc + 1) * sizeof(int));
    for (int j = 1; j <= ny; ++j) {
        int J = g->j0 + j;
        c->cy[j] = J / m;
        c->wy[j] = (double)(J % m) / m;
    }
    c->nseg = 0;
    for (int i = 1; i <= nx; ++i) {
        int I = g->i0 + i, cell = I / m;
        c->w1[i] = (double)(I % m) / m;
        c->w0[i] = 1.0 - c->w1[i];
        if (c->nseg == 0 || c->seg_c[c->nseg - 1] != cell) {
            c->seg_c[c->nseg] = cell;
            c->seg_lo[c->nseg] = i;
            ++c->nseg;
        }
        c->seg_hi[c->nseg - 1] = i;
    }
    /* restriction 4 + prolongation 4 per fine point */
    c->flops_fine = 8.0 * (double)(g->N - 1) * (g->N - 1);
    /* banded forward + backward substitution */
    c->flops_solve = 4.0 * (double)c->nc * c->bw;
}

void coarse_free(Coarse *c)
{
    free(c->L); free(c->rc); free(c->row); free(c->cy); free(c->wy);
    free(c->w0); free(c->w1); free(c->seg_c); free(c->seg_lo); free(c->seg_hi);
}

void coarse_begin(Coarse *c)
{
    memset(c->rc, 0, (size_t)(c->Nc + 1) * (c->Nc + 1) * sizeof(double));
}

/* rc += J^T restricted to fine row j: per coarse cell two weighted sums
 * (vectorisable), then split between coarse rows cy and cy+1. */
void coarse_restrict_row(Coarse *c, int j, const double *rj)
{
    const int Nc1 = c->Nc + 1;
    const double wy = c->wy[j];
    double *r0 = c->rc + (size_t)c->cy[j] * Nc1, *r1 = r0 + Nc1;
    const double *w0 = c->w0, *w1 = c->w1;
    for (int s = 0; s < c->nseg; ++s) {
        double s0 = 0.0, s1 = 0.0;
#pragma omp simd reduction(+:s0, s1)
        for (int i = c->seg_lo[s]; i <= c->seg_hi[s]; ++i) {
            s0 += w0[i] * rj[i];
            s1 += w1[i] * rj[i];
        }
        int k = c->seg_c[s];
        r0[k]     += (1.0 - wy) * s0;
        r1[k]     += wy * s0;
        r0[k + 1] += (1.0 - wy) * s1;
        r1[k + 1] += wy * s1;
    }
}

/* All-reduce the restricted residual, then solve on every rank. */
void coarse_solve(Coarse *c, const Grid *g, Timers *t)
{
    const int Nc = c->Nc, Nc1 = Nc + 1;
    double *rc = c->rc;
    double t0 = MPI_Wtime();
    if (g->P > 1 && g->reduce_on)
        MPI_Allreduce(MPI_IN_PLACE, rc, Nc1 * Nc1, MPI_DOUBLE, MPI_SUM, g->comm);
    double t1 = MPI_Wtime();

    /* compact the interior nodes in place, solve, expand back */
    double *b = rc;
    for (int J = 1; J < Nc; ++J)
        for (int I = 1; I < Nc; ++I)
            b[(I - 1) + (size_t)(J - 1) * c->n1] = rc[I + (size_t)J * Nc1];
    band_solve(c, b);
    for (int J = Nc - 1; J >= 1; --J)
        for (int I = Nc - 1; I >= 1; --I)
            rc[I + (size_t)J * Nc1] = b[(I - 1) + (size_t)(J - 1) * c->n1];
    for (int k = 0; k < Nc1; ++k) {                /* boundary nodes are zero */
        rc[k] = rc[k + (size_t)Nc * Nc1] = 0.0;
        rc[(size_t)k * Nc1] = rc[Nc + (size_t)k * Nc1] = 0.0;
    }
    double t2 = MPI_Wtime();
    t->coarse_comm += t1 - t0;
    t->coarse_solve += t2 - t1;
}

/* z_j += (J x) on fine row j. */
void coarse_prolong_row(const Coarse *c, int j, double *zj)
{
    const int Nc1 = c->Nc + 1;
    const double wy = c->wy[j];
    const double *x0 = c->rc + (size_t)c->cy[j] * Nc1, *x1 = x0 + Nc1;
    const double *w0 = c->w0, *w1 = c->w1;
    for (int s = 0; s < c->nseg; ++s) {
        int k = c->seg_c[s];
        double a = (1.0 - wy) * x0[k] + wy * x1[k];
        double b = (1.0 - wy) * x0[k + 1] + wy * x1[k + 1];
#pragma omp simd
        for (int i = c->seg_lo[s]; i <= c->seg_hi[s]; ++i)
            zj[i] += w0[i] * a + w1[i] * b;
    }
}

void coarse_apply(Coarse *c, const Grid *g, const double *r, double *z, Timers *t)
{
    double t0 = MPI_Wtime();
    coarse_begin(c);
    for (int j = 1; j <= g->ny; ++j) coarse_restrict_row(c, j, r + IDX(g, 0, j));
    double t1 = MPI_Wtime();
    coarse_solve(c, g, t);
    double t2 = MPI_Wtime();
    for (int j = 1; j <= g->ny; ++j) coarse_prolong_row(c, j, z + IDX(g, 0, j));
    t->coarse_rp += (t1 - t0) + (MPI_Wtime() - t2);
}
