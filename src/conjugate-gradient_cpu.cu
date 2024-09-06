#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int get_index(int *cords, int L, int d, int N) {
  for (int c = 0; c < d; c++) {
    int cord = cords[c];
    assert(cord < L + 1 && cord > -2);
    if (cord == -1 || cord == L) {
      return N;
    }
  }

  int ind = 0;
  for (int i = 0; i < d; i++) {
    ind += pow(L, i) * cords[i];
  }
  return ind;
}

int neighbour_index(int *cords, int direction, int amount, int L, int d,
                    int N) {
  int copy_cords[d];
  memcpy(copy_cords, cords, d * sizeof(int));
  copy_cords[direction] += amount;
  return get_index(copy_cords, L, d, N);
}

int *index_to_cords(int *cords, int index, int L, int d) {
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++) {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}

double *minus_laplace(double *ddf, double *u, int d, int L, int N) {
  for (int ind = 0; ind < N; ind++) {
    int cords[d];
    index_to_cords(cords, ind, L, d);
    double laplace_value = 0;
    for (int i = 0; i < d; i++) {
      laplace_value += -u[neighbour_index(cords, i, 1, L, d, N)] +
                       2 * u[neighbour_index(cords, i, 0, L, d, N)] -
                       u[neighbour_index(cords, i, -1, L, d, N)];
    }
    ddf[ind] = laplace_value;
  }
  return ddf;
}

double norm(double *x, int n) {
  double norm = 0;
  for (int i = 0; i < n; i++) {
    norm += x[i] * x[i];
  }
  return sqrt(norm);
}

double norm(const double *x, int n) {
  double norm = 0;
  for (int i = 0; i < n; i++) {
    norm += x[i] * x[i];
  }
  return sqrt(norm);
}

double inner_product(const double *x, const double *y, int n) {
  double product = 0;
  for (int i = 0; i < n; i++) {
    product += x[i] * y[i];
  }
  return product;
}

void print_matrix(double *A, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
}

double *allocate_field(int N) {
  double *r = (double *)calloc(N + 1, sizeof(double));
  if (r == NULL) {
    printf("Memory allocation failed");
    exit(1);
  }
  r[N] = 0;
  return r;
}

double conjugate_gradient(const double *b, double *x, int L, int d) {
  printf("calling cg");
  // Solve Ax = b
  // x is initial guess
  // x is overwritten
  // returns norm of residue
  int N = pow(L, d);
  double *r = allocate_field(N);
  minus_laplace(x, x, d, L, N);
  for (int i = 0; i < N; i++) {
    r[i] = b[i] - x[i];
  }
  double tol = 1e-6 * norm(b, N);
  // p = r;
  double *p = allocate_field(N);
  for (int i = 0; i < N; i++) {
    p[i] = r[i];
  }
  double *Ap = allocate_field(N);
  float dx = 2.0 / (L - 1);
  int n = 0;
  double rr = NAN;
  double residue = norm(r, N);
  while (residue > tol) {
    printf("iteration %d\n",n);
    

    minus_laplace(Ap, p, d, L, N);
    printf("inner_product(p, p, N): %f\n",inner_product(p, p, N));
    printf("inner_product(p, Ap, N): %f\n",inner_product(p, Ap, N));

    printf("array p: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", p[i]);
    }
    printf("\n");
    
    printf("array Ap: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", Ap[i]);
    }
    printf("\n");
    
    printf("array r: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", r[i]);
    }
    printf("\n");
    
    // double dx = 2.0 / (L - 1);
    // TODO Reuse variables
    rr = inner_product(r, r, N); // todo move this inner product to the beginning. You allready updated rr in the last line. This should be initialization.
    double alpha = rr / inner_product( p, Ap, N);
    for (int i = 0; i < N; i++) {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    double rr_new = inner_product(r, r, N);
    printf("rrnew : %f, alpha:  %f ", rr_new, alpha);
    double beta = rr_new / rr;
    for (int i = 0; i < N; i++) {
      p[i] = r[i] + beta * p[i];
    }
    printf("inner_product_gpu(p_old, Ap, N) should be 0: %f\n",inner_product(p, Ap, N));
    residue = sqrt(rr);
    rr = rr_new;
    printf("residue: %f at iteration: %i\n", residue, n);
    n++;
  }
  free(Ap);
  free(r);
  free(p);
  return residue;
}
// on the GPU this will be done with floats and the conversion will be done in
// the kernel
double preconditioner(double *b, double *x, int L, int d, double errtol) {
  // For testing
  // for (int i=0; i< pow(L,d); i++)
  // {
  //     x[i] = b[i];
  // }

  int N = pow(L, d);
  double *r = allocate_field(N);
  for (int i = 0; i < N; i++) {
    r[i] = b[i];
  }
  double tol = errtol * norm(b, N);
  // p = r;
  double *p = allocate_field(N);
  for (int i = 0; i < N; i++) {
    p[i] = r[i];
  }
  double *Ap = allocate_field(N);
  float dx = 2.0 / (L - 1);
  int i = 0;
  for (int k = 0; k < N; k++) {
    x[k] = 0;
  }
  double res = norm(r, N);
  assert(x[0] == 0);
  double rr = NAN;
  while (res > tol) {
    i++;
    minus_laplace(Ap, p, d, L, N);
    // double dx = 2.0 / (L - 1);
    // TODO Reuse variables
    rr = inner_product(r, r, N);
    double alpha = rr / inner_product(p, Ap, N);
    for (int i = 0; i < N; i++) {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    double rr_new = inner_product(r, r, N);
    double beta = rr_new / rr;
    rr = rr_new;
    for (int i = 0; i < N; i++) {
      p[i] = r[i] + beta * p[i];
    }
    res = norm(r, N);
    // printf("inner (b,x) = %g\n", inner_product(b,x,N));
    // printf("inner res: %g, i=%d \n" ,res,i);
  }
  printf("number of steps inner %d\n", i);
  free(Ap);
  free(r);
  free(p);
  return res;
}

double *preconditioned_cg(double *b, double *x, int L, int d) {
  int N = pow(L, d);
  assert(N > 0);
  double *r = allocate_field(N);
  double *Ax = allocate_field(N);
  minus_laplace(Ax, x, d, L, N);
  for (int i = 0; i < N; i++) {
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
  for (int i = 0; i < N; i++) {
    Minv_r[i] = p[i];
  }
  rMinvr = inner_product(r, Minv_r, N);
  double r_newMinvr_new;
  int i = 0;
  int maxitter = 1000;
  while (i < maxitter) {
    i++;
    minus_laplace(Ap, p, d, L, N);
    double alpha =
        rMinvr /
        inner_product(p, Ap, N);
    for (int i = 0; i < N; i++) {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }
    if (norm(r, N) < tol)
      break;
    preconditioner(r, Minv_r_new, L, d, 1e-3);
    r_newMinvr_new = inner_product(r, Minv_r_new, N);
    double beta = r_newMinvr_new / rMinvr;
    for (int i = 0; i < N; i++) {
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
