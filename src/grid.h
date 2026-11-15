#ifndef GRID_H
#define GRID_H

/*
 * 2D domain decomposition of the (N-1)^2 interior points of the unit square,
 * h = 1/N.  Global interior indices I, J run over 1..N-1 (x and y); I = 0, N
 * and J = 0, N are the Dirichlet boundary.
 *
 * Rank (px, py) of a Px x Py Cartesian grid owns the block
 *     I = i0+1 .. i0+nx,   J = j0+1 .. j0+ny
 * and stores it with one ghost layer, (nx+2) x (ny+2), x unit-stride:
 *     u[IDX(g,i,j)],  i = 0..nx+1, j = 0..ny+1,  global (i0+i, j0+j).
 * Ghosts on a physical boundary hold boundary data; the others are filled
 * by halo_exchange().
 */

#include <mpi.h>

typedef struct {
    int N;              /* h = 1/N */
    double h, ih2;      /* h and 1/h^2 */
    int P, rank;
    int Px, Py, px, py;
    int nx, ny;         /* local interior size */
    int i0, j0;         /* global index of local i = 0, j = 0 */
    int ld;             /* = nx + 2, leading dimension */
    int west, east, south, north;   /* neighbour ranks or MPI_PROC_NULL */
    MPI_Comm comm;
    double *buf;        /* 4*ny packing buffers for the x-direction columns */
    int halo_on, reduce_on;         /* ablation switches (default 1) */
} Grid;

#define IDX(g, i, j) ((size_t)(j) * (size_t)(g)->ld + (size_t)(i))

/* Per-rank wall-clock accumulators (seconds). */
typedef struct {
    double halo;        /* nearest-neighbour exchange (incl. waiting) */
    double reduce;      /* scalar all-reduces (incl. waiting) */
    double coarse_comm; /* all-reduce of the coarse residual */
    double coarse_solve;/* redundant coarse solve */
    double coarse_rp;   /* local restriction + prolongation */
    double setup;       /* solver setup before the iteration loop */
    double total;       /* iteration loop only (excludes setup) */
} Timers;

void   grid_init(Grid *g, int N, int Px, int Py, MPI_Comm comm);
void   grid_free(Grid *g);
double *grid_alloc(const Grid *g);   /* zeroed, (nx+2)*(ny+2) */

void   halo_exchange(const Grid *g, double *u, Timers *t);
double allreduce_max(const Grid *g, double x, Timers *t);
void   allreduce_sum(const Grid *g, double *x, int n, Timers *t);

/* Boundary / exact solution. */
double ue(double x, double y);
void   set_boundary(const Grid *g, double *u);           /* physical ghosts = ue */
void   set_bc_only(const Grid *g, double *ub);           /* ub: interior 0, ghosts ue */

/* Global max-norm of the interior of u (all ranks get the result). */
double norm_inf(const Grid *g, const double *u, Timers *t);
/* Global max |u - ue| over the interior. */
double err_vs_exact(const Grid *g, const double *u);

/* Gather the interior of u to rank 0 as an (N-1)x(N-1) array (x fastest).
 * Returns a malloc'd buffer on rank 0, NULL elsewhere. */
double *gather_interior(const Grid *g, const double *u);

#endif
