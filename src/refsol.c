/*
 * Exact solution u* of the discrete system A u = b (A = -L_h on the interior,
 * b = L_h u_b), used to measure the true algebraic error of the iterative
 * solvers.  A = (T (x) I + I (x) T)/h^2 with T = tridiag(-1, 2, -1); the
 * orthonormal sine matrix S_ik = sqrt(2/N) sin(pi i k / N) (S = S^T = S^-1)
 * diagonalises T with eigenvalues 4 sin^2(pi k / 2N).  Hence
 *     U* = S [ (S B S) ./ Lambda ] S,   Lambda_kl = (lam_k + lam_l)/h^2,
 * with U, B stored n x n, n = N-1.  Dense O(n^3), rank 0 only.
 */
#include "solver.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* C = S*B (left = 1) or C = B*S (left = 0); n x n, column-major. */
static void mul(int n, const double *S, const double *B, double *C, int left)
{
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
            double *c = C + (size_t)j * n;
            if (left) {
                double b = B[k + (size_t)j * n];
                const double *s = S + (size_t)k * n;
                for (int i = 0; i < n; ++i) c[i] += s[i] * b;
            } else {
                double s = S[k + (size_t)j * n];
                const double *b = B + (size_t)k * n;
                for (int i = 0; i < n; ++i) c[i] += b[i] * s;
            }
        }
}

double *ref_solution(const Grid *g)
{
    const int N = g->N, n = N - 1;
    double *U = malloc((size_t)n * n * sizeof(double));
    if (g->rank == 0) {
        double *S = malloc((size_t)n * n * sizeof(double));
        double *B = calloc((size_t)n * n, sizeof(double));
        double *T = malloc((size_t)n * n * sizeof(double));
        double *lam = malloc((size_t)n * sizeof(double));
        const double c = sqrt(2.0 / N), h = 1.0 / N, ih2 = (double)N * N;
        for (int k = 1; k <= n; ++k) {
            double sk = sin(M_PI * k / (2.0 * N));
            lam[k - 1] = 4.0 * sk * sk;
            for (int i = 1; i <= n; ++i)
                S[(i - 1) + (size_t)(k - 1) * n] = c * sin(M_PI * (double)((long)i * k % (2 * N)) / N);
        }
        /* b = L_h u_b: boundary neighbours only */
        for (int J = 1; J <= n; ++J)
            for (int I = 1; I <= n; ++I) {
                double v = 0.0;
                if (I == 1) v += ue(0.0, J * h);
                if (I == n) v += ue(1.0, J * h);
                if (J == 1) v += ue(I * h, 0.0);
                if (J == n) v += ue(I * h, 1.0);
                B[(I - 1) + (size_t)(J - 1) * n] = ih2 * v;
            }
        mul(n, S, B, T, 1);
        mul(n, S, T, B, 0);
        for (int l = 0; l < n; ++l)
            for (int k = 0; k < n; ++k)
                B[k + (size_t)l * n] /= ih2 * (lam[k] + lam[l]);
        mul(n, S, B, T, 1);
        mul(n, S, T, U, 0);
        free(S); free(B); free(T); free(lam);
    }
    /* broadcast in chunks (count is an int) */
    size_t tot = (size_t)n * n, chunk = 1u << 26;
    for (size_t off = 0; off < tot; off += chunk) {
        size_t cnt = tot - off < chunk ? tot - off : chunk;
        MPI_Bcast(U + off, (int)cnt, MPI_DOUBLE, 0, g->comm);
    }
    double *u = grid_alloc(g);
    for (int j = 1; j <= g->ny; ++j)
        for (int i = 1; i <= g->nx; ++i)
            u[IDX(g, i, j)] = U[(g->i0 + i - 1) + (size_t)(g->j0 + j - 1) * n];
    free(U);
    return u;
}
