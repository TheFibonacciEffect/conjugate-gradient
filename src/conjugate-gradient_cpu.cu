#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/**
 * @brief Calculates the index of a point in a multidimensional array.
 *
 * This function takes in the coordinates of a point in a multidimensional array
 * and calculates its index. The coordinates are specified as an array of
 * integers, where each integer represents the coordinate along a particular
 * dimension. The function assumes that the array has dimensions of size L in
 * each dimension, and that there are d dimensions in total. The total number of
 * elements in the array is given by N.
 *
 * @param cords The coordinates of the point in the multidimensional array.
 * @param L The size of each dimension in the array.
 * @param d The number of dimensions in the array.
 * @param N The total number of elements in the array.
 * @return The index of the point in the array.
 */
int get_index(int *cords, int L, int d, int N)
{
  for (int c = 0; c < d; c++)
  {
    int cord = cords[c];
    assert(cord < L + 1 && cord > -2);
    if (cord == -1 || cord == L)
    {
      return N;
    }
  }

  int ind = 0;
  for (int i = 0; i < d; i++)
  {
    ind += pow(L, i) * cords[i];
  }
  return ind;
}

/**
 * Calculates the index of a neighboring element in a multidimensional array.
 *
 * @param cords The coordinates of the current element.
 * @param direction The direction in which to move from the current element.
 * @param amount The amount by which to move in the specified direction.
 * @param L The size of each dimension of the array.
 * @param d The number of dimensions in the array.
 * @param N The total number of elements in the array.
 * @return The index of the neighboring element.
 */
int neighbour_index(int *cords, int direction, int amount, int L, int d,
                    int N)
{
  int copy_cords[d];
  memcpy(copy_cords, cords, d * sizeof(int));
  copy_cords[direction] += amount;
  return get_index(copy_cords, L, d, N);
}
/**
 * @brief Converts an index to coordinates in a multi-dimensional grid.
 *
 * This function takes an index, the size of the grid (L), and the number of
 * dimensions (d), and returns the corresponding coordinates in the grid. The
 * index should be within the range [0, pow(L, d)-1]. The function modifies the
 * `cords` array to store the computed coordinates.
 *
 * @param cords The array to store the computed coordinates.
 * @param index The index to convert to coordinates.
 * @param L The size of the grid.
 * @param d The number of dimensions.
 * @return The modified `cords` array.
 */
int *index_to_cords(int *cords, int index, int L, int d)
{
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++)
  {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}
/**
 * Calculates the discrete Laplacian of a given function on a grid using the
 * finite difference method.
 *
 * @param ddf - Pointer to the array where the Laplacian values will be stored.
 * @param u - Pointer to the array containing the function values on the grid.
 * @param d - The dimensionality of the grid.
 * @param L - The size of each dimension of the grid.
 * @param N - The total number of grid points.
 * @return Pointer to the array containing the Laplacian values.
 */
double *minus_laplace(double *ddf, double *u, int d, int L, int N)
{
  for (int ind = 0; ind < N; ind++)
  {
    int cords[d];
    index_to_cords(cords, ind, L, d);
    double laplace_value = 0;
    for (int i = 0; i < d; i++)
    {
      laplace_value += -u[neighbour_index(cords, i, 1, L, d, N)] +
                       2 * u[neighbour_index(cords, i, 0, L, d, N)] -
                       u[neighbour_index(cords, i, -1, L, d, N)];
    }
    ddf[ind] = laplace_value;
  }
  return ddf;
}

/**
 * Calculates the Euclidean norm of a given vector.
 *
 * @param x - Pointer to the vector.
 * @param n - The length of the vector.
 * @return The Euclidean norm of the vector.
 */
double norm(double *x, int n)
{
  double norm = 0;
  for (int i = 0; i < n; i++)
  {
    norm += x[i] * x[i];
  }
  return sqrt(norm);
}

double norm(const double *x, int n)
{
  double norm = 0;
  for (int i = 0; i < n; i++)
  {
    norm += x[i] * x[i];
  }
  return sqrt(norm);
}

/**
 * Calculates the inner product of two arrays.
 *
 * @param x - The first array.
 * @param y - The second array.
 * @param n - The size of the arrays.
 * @return The inner product of the two arrays.
 */
double inner_product(const double *x, const double *y, int n)
{
  double product = 0;
  for (int i = 0; i < n; i++)
  {
    product += x[i] * y[i];
  }
  return product;
}

/**
 * @brief Prints a matrix.
 *
 * This function prints a matrix represented by a 1D array of doubles.
 *
 * @param A Pointer to the matrix array.
 * @param n The size of the matrix (n x n).
 */
void print_matrix(double *A, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
}

/**
 * @brief Allocates memory for a field of doubles. The Nth element is set to
 * zero and used for the boundary condition.
 *
 * This function allocates memory for a field of doubles with a size of N+1. The
 * memory is initialized to zero, and a pointer to the allocated memory is
 * returned.
 *
 * @param N The size of the field.
 * @return A pointer to the allocated memory.
 */
double *allocate_field(int N)
{
  double *r = (double *)calloc(N + 1, sizeof(double));
  if (r == NULL)
  {
    printf("Memory allocation failed");
    exit(1);
  }
  r[N] = 0;
  return r;
}

/**
 * @brief Performs the Conjugate Gradient method to solve the laplacian.
 *
 * This function uses the Conjugate Gradient method to solve a linear system of
 * equations represented by the equation Ax = b, where A is the laplacian
 * matrix, b is the right-hand side vector, and x is the solution vector.
 *
 * @param b Pointer to the right-hand side vector.
 * @param x Pointer to the solution vector.
 * @param L The size of the grid in one dimension.
 * @param d The dimension of the grid.
 * @return The residue of the solution vector.
 */
double conjugate_gradient(const double *b, double *x, int L, int d)
{
  int N = pow(L, d);
  double *r = allocate_field(N);
  minus_laplace(x, x, d, L, N);
  for (int i = 0; i < N; i++)
  {
    r[i] = b[i] - x[i];
  }
  double tol = 1e-6 * norm(b, N);
  // p = r;
  double *p = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    p[i] = r[i];
  }
  double *Ap = allocate_field(N);
  float dx = 2.0 / (L - 1);
  int n = 0;
  double rr = inner_product(r, r, N);
  double residue = norm(r, N);
  while (residue > tol)
  {

    minus_laplace(Ap, p, d, L, N);
    double alpha = rr / inner_product(p, Ap, N);
    for (int i = 0; i < N; i++)
    {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    double rr_new = inner_product(r, r, N);
    double beta = rr_new / rr;
    for (int i = 0; i < N; i++)
    {
      p[i] = r[i] + beta * p[i];
    }
    residue = sqrt(rr);
    rr = rr_new;
    n++;
  }
  free(Ap);
  free(r);
  free(p);
  return residue;
}

/**
 * Calculates the conjugate gradient method as a preconditioner for the
 * conjugate gradient method.
 *
 * @param b The input vector.
 * @param x The output vector.
 * @param L The size of the grid.
 * @param d The dimension of the grid.
 * @param errtol The error tolerance.
 * @return The residual of the preconditioner.
 */
double preconditioner(double *b, double *x, int L, int d, double errtol)
{
  int N = pow(L, d);
  double *r = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    r[i] = b[i];
  }
  double tol = errtol * norm(b, N);
  // p = r;
  double *p = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    p[i] = r[i];
  }
  double *Ap = allocate_field(N);
  float dx = 2.0 / (L - 1);
  int i = 0;
  for (int k = 0; k < N; k++)
  {
    x[k] = 0;
  }
  double res = norm(r, N);
  assert(x[0] == 0);
  double rr = NAN;
  while (res > tol)
  {
    i++;
    minus_laplace(Ap, p, d, L, N);
    rr = inner_product(r, r, N);
    double alpha = rr / inner_product(p, Ap, N);
    for (int i = 0; i < N; i++)
    {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    double rr_new = inner_product(r, r, N);
    double beta = rr_new / rr;
    rr = rr_new;
    for (int i = 0; i < N; i++)
    {
      p[i] = r[i] + beta * p[i];
    }
    res = norm(r, N);
  }
  printf("number of steps inner %d\n", i);
  free(Ap);
  free(r);
  free(p);
  return res;
}
/**
 * Solves a linear system using the Preconditioned Conjugate Gradient method.
 *
 * @param b The right-hand side vector of the linear system.
 * @param x The initial guess for the solution vector.
 * @param L The size of the grid in each dimension.
 * @param d The number of dimensions.
 * @return The solution vector.
 */
double *preconditioned_cg(double *b, double *x, int L, int d)
{
  int N = pow(L, d);
  assert(N > 0);
  double *r = allocate_field(N);
  double *Ax = allocate_field(N);
  minus_laplace(Ax, x, d, L, N);
  for (int i = 0; i < N; i++)
  {
    r[i] = b[i] - Ax[i];
  }
  double tol = 1e-8 * norm(b, N);
  // p = M^-1r;
  double *p = allocate_field(N);
  preconditioner(r, p, L, d, 1e-3);
  float dx = 2.0 / (L - 1);
  double *Minv_r_new = Ax;
  double *Minv_r = Ax;
  double *Ap = Ax;
  double rMinvr;
  for (int i = 0; i < N; i++)
  {
    Minv_r[i] = p[i];
  }
  rMinvr = inner_product(r, Minv_r, N);
  double r_newMinvr_new;
  int i = 0;
  int maxitter = 1000;
  while (i < maxitter)
  {
    i++;
    minus_laplace(Ap, p, d, L, N);
    double alpha =
        rMinvr /
        inner_product(p, Ap, N);
    for (int i = 0; i < N; i++)
    {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    if (norm(r, N) < tol)
      break;
    preconditioner(r, Minv_r_new, L, d, 1e-3);
    r_newMinvr_new = inner_product(r, Minv_r_new, N);
    double beta = r_newMinvr_new / rMinvr;
    for (int i = 0; i < N; i++)
    {
      p[i] = Minv_r_new[i] + beta * p[i];
    }
    rMinvr = r_newMinvr_new;
  }
  printf("number of steps (outer) %i\n", i);
  free(Ax);
  // free(Minv_r);
  // free(Minv_r_new);
  free(r);
  free(p);
  return x;
}

double *random_array(double *r, int L, int d, int N)
{
  // double* r = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    r[i] = (double)rand() / (double)RAND_MAX;
  }
  return r;
}

