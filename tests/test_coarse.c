/*
 * Unit test for the coarse-grid correction (serial, small N):
 *   1. builds J (bilinear interpolation) and A densely and checks that the
 *      9-point Galerkin stencil used by coarse.c equals J^T A J entry by entry;
 *   2. checks coarse_apply(r) == J Ac^{-1} J^T r against a dense solve.
 * Usage: test_coarse   (exit status 0 on success)
 */
#include "../src/coarse.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double hat(double t) { t = fabs(t); return t < 1 ? 1 - t : 0; }

/* dense Gaussian elimination, A n x n row-major (destroyed), b in/out */
static void dense_solve(int n, double *A, double *b)
{
    for (int k = 0; k < n; ++k)
        for (int i = k + 1; i < n; ++i) {
            double l = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j) A[i * n + j] -= l * A[k * n + j];
            b[i] -= l * b[k];
        }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
}

static int run(int N, int Nc)
{
    Grid g;
    grid_init(&g, N, 1, 1, MPI_COMM_WORLD);
    Coarse c;
    coarse_init(&c, &g, Nc);
    const int n1 = N - 1, nf = n1 * n1, k1 = Nc - 1, ncs = k1 * k1, m = N / Nc;

    /* J: nf x ncs, A: nf x nf (dense, row-major) */
    double *J = calloc((size_t)nf * ncs, sizeof(double));
    double *A = calloc((size_t)nf * nf, sizeof(double));
    for (int J_ = 1; J_ <= n1; ++J_)
        for (int I = 1; I <= n1; ++I) {
            int f = (I - 1) + (J_ - 1) * n1;
            for (int cJ = 1; cJ < Nc; ++cJ)
                for (int cI = 1; cI < Nc; ++cI)
                    J[(size_t)f * ncs + (cI - 1) + (cJ - 1) * k1] =
                        hat((double)(I - cI * m) / m) * hat((double)(J_ - cJ * m) / m);
            A[(size_t)f * nf + f] = 4 * g.ih2;
            if (I > 1)  A[(size_t)f * nf + f - 1]  = -g.ih2;
            if (I < n1) A[(size_t)f * nf + f + 1]  = -g.ih2;
            if (J_ > 1)  A[(size_t)f * nf + f - n1] = -g.ih2;
            if (J_ < n1) A[(size_t)f * nf + f + n1] = -g.ih2;
        }
    /* Ac = J^T A J */
    double *AJ = calloc((size_t)nf * ncs, sizeof(double));
    double *Ac = calloc((size_t)ncs * ncs, sizeof(double));
    for (int i = 0; i < nf; ++i)
        for (int k = 0; k < nf; ++k) {
            double a = A[(size_t)i * nf + k];
            if (a != 0) for (int j = 0; j < ncs; ++j) AJ[(size_t)i * ncs + j] += a * J[(size_t)k * ncs + j];
        }
    for (int i = 0; i < nf; ++i)
        for (int a = 0; a < ncs; ++a) {
            double ja = J[(size_t)i * ncs + a];
            if (ja != 0) for (int b = 0; b < ncs; ++b) Ac[(size_t)a * ncs + b] += ja * AJ[(size_t)i * ncs + b];
        }
    double err_st = 0, scale = 0;
    for (int a = 0; a < ncs; ++a)
        for (int b = 0; b < ncs; ++b) {
            int dI = b % k1 - a % k1, dJ = b / k1 - a / k1;
            double want = (abs(dI) <= 1 && abs(dJ) <= 1) ? c.stencil[dJ + 1][dI + 1] : 0.0;
            err_st = fmax(err_st, fabs(Ac[(size_t)a * ncs + b] - want));
            scale = fmax(scale, fabs(Ac[(size_t)a * ncs + b]));
        }

    /* coarse_apply vs dense J Ac^{-1} J^T r */
    double *r = grid_alloc(&g), *z = grid_alloc(&g);
    double *rv = malloc(nf * sizeof(double)), *rc = calloc(ncs, sizeof(double));
    for (int J_ = 1; J_ <= n1; ++J_)
        for (int I = 1; I <= n1; ++I)
            r[IDX(&g, I, J_)] = rv[(I - 1) + (J_ - 1) * n1] = sin(0.37 * I * I + 1.3 * J_);
    Timers t = {0};
    coarse_apply(&c, &g, r, z, &t);
    for (int i = 0; i < nf; ++i)
        for (int a = 0; a < ncs; ++a) rc[a] += J[(size_t)i * ncs + a] * rv[i];
    dense_solve(ncs, Ac, rc);
    double err_ap = 0, zs = 0;
    for (int J_ = 1; J_ <= n1; ++J_)
        for (int I = 1; I <= n1; ++I) {
            int f = (I - 1) + (J_ - 1) * n1;
            double want = 0;
            for (int a = 0; a < ncs; ++a) want += J[(size_t)f * ncs + a] * rc[a];
            err_ap = fmax(err_ap, fabs(z[IDX(&g, I, J_)] - want));
            zs = fmax(zs, fabs(want));
        }
    printf("N=%3d Nc=%2d m=%2d  stencil [%.4f %.4f %.4f]*h^2  |Ac-J^TAJ|/|Ac|=%.2e  "
           "|z-z_dense|/|z|=%.2e\n", N, Nc, m, c.stencil[1][1] / g.ih2,
           c.stencil[1][2] / g.ih2, c.stencil[2][2] / g.ih2, err_st / scale, err_ap / zs);
    int ok = err_st / scale < 1e-13 && err_ap / zs < 1e-11;
    free(J); free(A); free(AJ); free(Ac); free(r); free(z); free(rv); free(rc);
    coarse_free(&c);
    grid_free(&g);
    return ok;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int ok = 1;
    ok &= run(12, 2);
    ok &= run(12, 3);
    ok &= run(16, 4);
    ok &= run(24, 6);
    ok &= run(30, 5);
    ok &= run(32, 8);
    printf(ok ? "PASS\n" : "FAIL\n");
    MPI_Finalize();
    return !ok;
}
