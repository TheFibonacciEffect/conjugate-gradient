// https://www.gnu.org/software/gsl/doc/html/splinalg.html#examples
// this solves poisson's equation with GMRES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_splinalg.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_splinalg.h>

void construct_sparse_matrix(gsl_spmatrix *A, size_t n, double h);
void construct_rhs_vector(gsl_vector *f, size_t n, double h);
void solve_system(gsl_spmatrix *C, gsl_vector *f, gsl_vector *u, size_t n, double tol, size_t max_iter);
void print_solution(gsl_vector *u, size_t n, double h);

int main()
{
  const size_t N = 100;                       /* number of grid points */
  const size_t n = N - 2;                     /* subtract 2 to exclude boundaries */
  const double h = 1.0 / (N - 1.0);           /* grid spacing */
  gsl_spmatrix *A = gsl_spmatrix_alloc(n ,n); /* triplet format */
  gsl_spmatrix *C;                            /* compressed format */
  gsl_vector *f = gsl_vector_alloc(n);        /* right hand side vector */
  gsl_vector *u = gsl_vector_alloc(n);        /* solution vector */

  /* construct the sparse matrix for the finite difference equation */
  construct_sparse_matrix(A, n, h);

  /* construct right hand side vector */
  construct_rhs_vector(f, n, h);

  /* convert to compressed column format */
  C = gsl_spmatrix_ccs(A);

  /* now solve the system with the GMRES iterative solver */
  solve_system(C, f, u, n, 1.0e-6, 10);

  /* output solution */
  print_solution(u, n, h);

  gsl_spmatrix_free(A);
  gsl_spmatrix_free(C);
  gsl_vector_free(f);
  gsl_vector_free(u);

  return 0;
}

void construct_sparse_matrix(gsl_spmatrix *A, size_t n, double h)
{
  size_t i;

  /* construct first row */
  gsl_spmatrix_set(A, 0, 0, -2.0);
  gsl_spmatrix_set(A, 0, 1, 1.0);

  /* construct rows [1:n-2] */
  for (i = 1; i < n - 1; ++i)
  {
    gsl_spmatrix_set(A, i, i + 1, 1.0);
    gsl_spmatrix_set(A, i, i, -2.0);
    gsl_spmatrix_set(A, i, i - 1, 1.0);
  }

  /* construct last row */
  gsl_spmatrix_set(A, n - 1, n - 1, -2.0);
  gsl_spmatrix_set(A, n - 1, n - 2, 1.0);

  /* scale by h^2 */
  gsl_spmatrix_scale(A, 1.0 / (h * h));
}

void construct_rhs_vector(gsl_vector *f, size_t n, double h)
{
  size_t i;

  for (i = 0; i < n; ++i)
  {
    double xi = (i + 1) * h;
    double fi = -M_PI * M_PI * sin(M_PI * xi);
    gsl_vector_set(f, i, fi);
  }
}

void solve_system(gsl_spmatrix *C, gsl_vector *f, gsl_vector *u, size_t n, double tol, size_t max_iter)
{
  const gsl_splinalg_itersolve_type *T = gsl_splinalg_itersolve_gmres;
  gsl_splinalg_itersolve *work = gsl_splinalg_itersolve_alloc(T, n, 0);
  size_t iter = 0;
  double residual;
  int status;

  /* initial guess u = 0 */
  gsl_vector_set_zero(u);

  /* solve the system A u = f */
  do
  {
    status = gsl_splinalg_itersolve_iterate(C, f, tol, u, work);

    /* print out residual norm ||A*u - f|| */
    residual = gsl_splinalg_itersolve_normr(work);
    fprintf(stderr, "iter %zu residual = %.12e\n", iter, residual);

    if (status == GSL_SUCCESS)
      fprintf(stderr, "Converged\n");
  } while (status == GSL_CONTINUE && ++iter < max_iter);

  gsl_splinalg_itersolve_free(work);
}

void print_solution(gsl_vector *u, size_t n, double h)
{
  size_t i;

  for (i = 0; i < n; ++i)
  {
    double xi = (i + 1) * h;
    double u_exact = sin(M_PI * xi);
    double u_gsl = gsl_vector_get(u, i);

    printf("%f %.12e %.12e\n", xi, u_gsl, u_exact);
  }
}
