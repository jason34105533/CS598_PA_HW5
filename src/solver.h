#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"

#define MAXD 16

typedef struct {
    int maxit;          /* iteration cap */
    double tol;         /* relative residual tolerance; <= 0: run exactly maxit */
    int hist;           /* print a HIST line every hist iterations (0 = off) */
    const double *uref; /* exact discrete solution (interior, grid layout) or NULL */
    double refnorm;     /* ||uref||_inf */
    int guess_rand;     /* 1: pseudo-random initial interior guess */
    int Nc;             /* PCG coarse grid: Nc intervals, H = 1/Nc (0 = off) */
} Opts;

typedef struct {
    int iters;          /* iterations performed */
    double relres;      /* final relative residual (norm of the method) */
    double relerr;      /* final ||u - uref||_inf / ||uref||_inf, or -1 */
    double flops;       /* global flop count of the iteration loop */
    int kres[MAXD + 1]; /* first k with relres <= 10^-d (-1: never) */
    int kerr[MAXD + 1]; /* first k with relerr <= 10^-d (-1: never / no ref) */
    Timers t;
} Stats;

/* Jacobi on the full grid function u (ghosts carry boundary data).
 * Residual monitored in the max norm: ||r_k||_inf / ||r_0||_inf. */
void jacobi_solve(const Grid *g, double *u, const Opts *o, Stats *s);

/* (Jacobi-)preconditioned CG for A u0 = b, b = -A_grid u_b, A = -L_h.
 * On return u holds u0 + u_b (same layout as jacobi_solve).
 * Residual monitored in the 2 norm: ||r_k||_2 / ||r_0||_2. */
void pcg_solve(const Grid *g, double *u, const Opts *o, Stats *s);

/* Deterministic pseudo-random value in [-1, 1] for global point (I, J). */
double guess_value(int I, int J);

/* Exact solution of the discrete system on rank 0 via a dense sine
 * transform (O(N^3)), broadcast and returned in grid layout. */
double *ref_solution(const Grid *g);

#endif
