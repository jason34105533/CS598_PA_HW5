#include "grid.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void decomp_1d(int n, int P, int r, int *nloc, int *off)
{
    int base = n / P, rem = n % P;
    *nloc = base + (r < rem);
    *off  = r * base + (r < rem ? r : rem);
}

void grid_init(Grid *g, int N, int Px, int Py, MPI_Comm comm)
{
    memset(g, 0, sizeof *g);
    g->N = N;
    g->h = 1.0 / N;
    g->ih2 = (double)N * N;
    MPI_Comm_size(comm, &g->P);

    int dims[2] = {Px, Py}, periods[2] = {0, 0};
    if (Px <= 0 || Py <= 0) { dims[0] = dims[1] = 0; }
    MPI_Dims_create(g->P, 2, dims);
    MPI_Cart_create(comm, 2, dims, periods, 0, &g->comm);
    MPI_Comm_rank(g->comm, &g->rank);
    int coords[2];
    MPI_Cart_coords(g->comm, g->rank, 2, coords);
    g->Px = dims[0]; g->Py = dims[1];
    g->px = coords[0]; g->py = coords[1];
    MPI_Cart_shift(g->comm, 0, 1, &g->west, &g->east);
    MPI_Cart_shift(g->comm, 1, 1, &g->south, &g->north);

    decomp_1d(N - 1, g->Px, g->px, &g->nx, &g->i0);
    decomp_1d(N - 1, g->Py, g->py, &g->ny, &g->j0);
    g->ld = g->nx + 2;
    g->buf = malloc(4 * (size_t)(g->ny > 0 ? g->ny : 1) * sizeof(double));
    g->halo_on = g->reduce_on = 1;
}

void grid_free(Grid *g)
{
    free(g->buf);
    MPI_Comm_free(&g->comm);
}

double *grid_alloc(const Grid *g)
{
    size_t n = (size_t)(g->nx + 2) * (g->ny + 2);
    double *u = NULL;
    if (posix_memalign((void **)&u, 64, n * sizeof(double))) abort();
    memset(u, 0, n * sizeof(double));
    return u;
}

/*
 * Exchange one ghost layer with the four neighbours.  Rows (y-direction
 * neighbours) are contiguous and sent in place; columns are packed.
 * MPI_PROC_NULL makes the physical-boundary sides no-ops, so those ghosts
 * keep whatever boundary data the caller put there.
 */
void halo_exchange(const Grid *g, double *u, Timers *t)
{
    if (!g->halo_on || g->P == 1) return;
    double t0 = MPI_Wtime();
    const int nx = g->nx, ny = g->ny;
    double *sw = g->buf, *se = sw + ny, *rw = se + ny, *re = rw + ny;
    MPI_Request req[8];

    MPI_Irecv(&u[IDX(g, 1, 0)],      nx, MPI_DOUBLE, g->south, 0, g->comm, &req[0]);
    MPI_Irecv(&u[IDX(g, 1, ny + 1)], nx, MPI_DOUBLE, g->north, 1, g->comm, &req[1]);
    MPI_Irecv(rw, ny, MPI_DOUBLE, g->west, 2, g->comm, &req[2]);
    MPI_Irecv(re, ny, MPI_DOUBLE, g->east, 3, g->comm, &req[3]);

    if (g->west != MPI_PROC_NULL)
        for (int j = 1; j <= ny; ++j) sw[j - 1] = u[IDX(g, 1, j)];
    if (g->east != MPI_PROC_NULL)
        for (int j = 1; j <= ny; ++j) se[j - 1] = u[IDX(g, nx, j)];

    /* tag = direction of travel as seen by the receiver */
    MPI_Isend(&u[IDX(g, 1, ny)], nx, MPI_DOUBLE, g->north, 0, g->comm, &req[4]);
    MPI_Isend(&u[IDX(g, 1, 1)],  nx, MPI_DOUBLE, g->south, 1, g->comm, &req[5]);
    MPI_Isend(se, ny, MPI_DOUBLE, g->east, 2, g->comm, &req[6]);
    MPI_Isend(sw, ny, MPI_DOUBLE, g->west, 3, g->comm, &req[7]);
    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

    if (g->west != MPI_PROC_NULL)
        for (int j = 1; j <= ny; ++j) u[IDX(g, 0, j)] = rw[j - 1];
    if (g->east != MPI_PROC_NULL)
        for (int j = 1; j <= ny; ++j) u[IDX(g, nx + 1, j)] = re[j - 1];
    t->halo += MPI_Wtime() - t0;
}

double allreduce_max(const Grid *g, double x, Timers *t)
{
    if (!g->reduce_on || g->P == 1) return x;
    double t0 = MPI_Wtime(), y;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    t->reduce += MPI_Wtime() - t0;
    return y;
}

void allreduce_sum(const Grid *g, double *x, int n, Timers *t)
{
    if (!g->reduce_on || g->P == 1) return;
    double t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, g->comm);
    t->reduce += MPI_Wtime() - t0;
}

double ue(double x, double y)
{
    return sin(2.0 * x) * exp(-2.0 * y);
}

void set_boundary(const Grid *g, double *u)
{
    const double h = g->h;
    if (g->west == MPI_PROC_NULL)
        for (int j = 0; j <= g->ny + 1; ++j) u[IDX(g, 0, j)] = ue(0.0, (g->j0 + j) * h);
    if (g->east == MPI_PROC_NULL)
        for (int j = 0; j <= g->ny + 1; ++j) u[IDX(g, g->nx + 1, j)] = ue(1.0, (g->j0 + j) * h);
    if (g->south == MPI_PROC_NULL)
        for (int i = 0; i <= g->nx + 1; ++i) u[IDX(g, i, 0)] = ue((g->i0 + i) * h, 0.0);
    if (g->north == MPI_PROC_NULL)
        for (int i = 0; i <= g->nx + 1; ++i) u[IDX(g, i, g->ny + 1)] = ue((g->i0 + i) * h, 1.0);
}

void set_bc_only(const Grid *g, double *ub)
{
    memset(ub, 0, (size_t)(g->nx + 2) * (g->ny + 2) * sizeof(double));
    set_boundary(g, ub);
}

double norm_inf(const Grid *g, const double *u, Timers *t)
{
    double m = 0.0;
    for (int j = 1; j <= g->ny; ++j)
        for (int i = 1; i <= g->nx; ++i) m = fmax(m, fabs(u[IDX(g, i, j)]));
    if (t) return allreduce_max(g, m, t);
    double y;
    MPI_Allreduce(&m, &y, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    return y;
}

double err_vs_exact(const Grid *g, const double *u)
{
    double m = 0.0;
    for (int j = 1; j <= g->ny; ++j)
        for (int i = 1; i <= g->nx; ++i)
            m = fmax(m, fabs(u[IDX(g, i, j)] - ue((g->i0 + i) * g->h, (g->j0 + j) * g->h)));
    double y;
    MPI_Allreduce(&m, &y, 1, MPI_DOUBLE, MPI_MAX, g->comm);
    return y;
}

double *gather_interior(const Grid *g, const double *u)
{
    const int n = g->N - 1;
    int hdr[4] = {g->i0, g->j0, g->nx, g->ny};
    size_t cnt = (size_t)g->nx * g->ny;
    double *loc = malloc((cnt ? cnt : 1) * sizeof(double));
    for (int j = 0; j < g->ny; ++j)
        for (int i = 0; i < g->nx; ++i) loc[i + (size_t)j * g->nx] = u[IDX(g, i + 1, j + 1)];

    double *glob = NULL;
    if (g->rank == 0) {
        glob = malloc((size_t)n * n * sizeof(double));
        for (int r = 0; r < g->P; ++r) {
            int hr[4];
            double *b;
            if (r == 0) {
                memcpy(hr, hdr, sizeof hr);
                b = loc;
            } else {
                MPI_Recv(hr, 4, MPI_INT, r, 10, g->comm, MPI_STATUS_IGNORE);
                b = malloc(((size_t)hr[2] * hr[3] + 1) * sizeof(double));
                MPI_Recv(b, hr[2] * hr[3], MPI_DOUBLE, r, 11, g->comm, MPI_STATUS_IGNORE);
            }
            for (int j = 0; j < hr[3]; ++j)
                for (int i = 0; i < hr[2]; ++i)
                    glob[(hr[0] + i) + (size_t)(hr[1] + j) * n] = b[i + (size_t)j * hr[2]];
            if (r) free(b);
        }
    } else {
        MPI_Send(hdr, 4, MPI_INT, 0, 10, g->comm);
        MPI_Send(loc, (int)cnt, MPI_DOUBLE, 0, 11, g->comm);
    }
    free(loc);
    return glob;
}
