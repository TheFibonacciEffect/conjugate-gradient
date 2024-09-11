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
  int N = 100;
  int L = 10;
  int d = 2;

  // TODO compare laplace to cpu version

  // find the smallest and largest eigenvalue
  int nblocks = N;
  int nthreads = 1;
  // allocate an array
  float *z = cuda_allocate_field(N);
  float *q = cuda_allocate_field(N);
  // fill it with random data
  random_array(q, L, d, N);
  float lambda_min = 0;
  int itterations = 100;
  for (int i = 0; i < itterations; i++)
  {
    // z = Ainv q
    conjugate_gradient_gpu(q,z,L,d);
    // q = z/norm(z) 
    float nz = norm(z,N);
    for (int j = 0; j < N; j++)
    {
      q[j] = z[j]/nz;
    }
    lambda_min = inner_product_gpu(q,z,N);
    printf("lambda min %d %f\n",i, lambda_min);
  }
  
  random_array(q, L, d, N);
  float lambda_max = 0;
  for (int i = 0; i < itterations; i++)
  {
    // z = A q
    laplace_gpu<<<nblocks, nthreads>>>(z, q, d, L, N, 0);
    lambda_max = inner_product_gpu(q,z,N);
    // q = z/norm(z)
    float nz = norm(z,N);
    for (int j = 0; j < N; j++)
    {
      q[j] = z[j]/nz;
    }
    printf("lambda max %d %f\n", i, lambda_max);
  }

  cudaFree(q);
  cudaFree(z);
}
