#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>

// #include <math.h>
// #include "interleave.cuh"
#include "common.h"
#include "conjugate-gradient_gpu.cuh"
#include "conjugate-gradient_cpu.h"

void cg_ones(int L, int d, int N)
{
  // allocate an array
  double dx = 2.0 / (L - 1);
  double *x = allocate_field(N);
  double *b = allocate_field(N);
  // fill it with random data
  // calculate b
  b = minus_laplace(b, x, d, L, N);
  // initial guess
  for (int i = 0; i < N; i++)
  {
    b[i] = 1;
    x[i] = 0;
  }
  double *x0 = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  conjugate_gradient(b, x0, L, d);
}

int main()
{

  // TODO compare laplace to cpu version

  // find the smallest and largest eigenvalue
  // int nblocks = 1000000;
  int nblocks = 100000;
  int nthreads = 312;
  int N = nblocks*nthreads;
  int L = N;
  int d = 1;
  // allocate an array
  float *z = cuda_allocate_field(N);
  float *q = cuda_allocate_field(N);
  // fill it with random data
  random_array(q, L, d, N);
  conjugate_gradient_gpu(q,z,L,d);
  cudaFree(q);
  cudaFree(z);
}
