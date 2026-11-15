#ifndef COARSE_H
#define COARSE_H

#include "grid.h"

/*
 * Two-level additive coarse correction  z += J Ac^{-1} J^T r.
 *
 * Coarse grid: Nc intervals, H = 1/Nc = m h (N must be divisible by Nc),
 * coarse interior nodes (cI, cJ) = 1..Nc-1 sit on fine points (cI m, cJ m).
 * J = bilinear interpolation (tensor product of 1D hat functions), so
 * Ac = J^T A J is a constant 9-point stencil, computed once.  Ac is factored
 * redundantly on every rank with a banded Cholesky (half-bandwidth Nc).
 *
 * Per application: local restriction J^T r into a (Nc+1)^2 array, one
 * MPI_Allreduce(SUM) of that array, a redundant banded solve on each rank,
 * and local prolongation added into z.
 */

typedef struct {
    int Nc, m, n1, nc, bw;  /* n1 = Nc-1, nc = n1^2, bw = n1 + 1 */
    double *L;              /* banded Cholesky factor, rows of length bw+1 */
    double *rc;             /* (Nc+1)^2 coarse work array */
    int *cy;                /* per local j: coarse cell index */
    double *wy;             /* per local j: weight of the upper coarse node */
    double *w0, *w1;        /* per local i: weights of the left/right node */
    int nseg;               /* local i range split into coarse cells: */
    int *seg_c, *seg_lo, *seg_hi;   /* cell index, first and last local i */
    double *row;            /* (Nc+1) scratch */
    double stencil[3][3];   /* Ac stencil [dJ+1][dI+1] */
    double flops_fine;      /* flops per application, local fine-grid part */
    double flops_solve;     /* flops per application, coarse solve */
} Coarse;

void coarse_init(Coarse *c, const Grid *g, int Nc);
void coarse_free(Coarse *c);
/* z += J Ac^{-1} J^T r (r, z in grid layout). */
void coarse_apply(Coarse *c, const Grid *g, const double *r, double *z, Timers *t);

/* The same in pieces, so the fine-grid parts can be fused into other sweeps:
 *   coarse_begin();  coarse_restrict_row(j, r_j) for every local j;
 *   coarse_solve();  coarse_prolong_row(j, z_j) for every local j.
 * r_j, z_j point at element i = 0 of local row j. */
void coarse_begin(Coarse *c);
void coarse_restrict_row(Coarse *c, int j, const double *rj);
void coarse_solve(Coarse *c, const Grid *g, Timers *t);
void coarse_prolong_row(const Coarse *c, int j, double *zj);

#endif
