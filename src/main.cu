#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
// #include <math.h>
// #include "interleave.cuh"
#include "common.h"
#include "gpu_utils.cuh"
#include "conjugate-gradient_cpu.h"


void cg_ones(int L, int d, int N) {
  // allocate an array
  double dx = 2.0 / (L - 1);
  double *x = allocate_field(N);
  double *b = allocate_field(N);
  // fill it with random data
  // calculate b
  b = minus_laplace(b, x, d, L, N);
  // initial guess
  for (int i = 0; i<N; i++)
  {
    b[i] = 1;
    x[i] = 0;
  }
  double *x0 = allocate_field(N);
  for (int i = 0; i < N; i++) {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  conjugate_gradient(b, x0, L, d);
}

int main()
{
  int N = 100;
  int L = N;
  int d = 1;
  // run on cpu
  // for demonstration
  cg_ones(L,d,N);

  // printf("%d\n",index_to_cords(10,3,2)); // 3x3x3 cube 
  printf("%d\n",neighbour_index_gpu(0,1,1,3,3,3*3*3,0)); 
  printf("%d\n",neighbour_index_gpu(3,1,1,3,3,3*3*3,0));
  printf("%d\n",neighbour_index_gpu(6,1,1,3,3,3*3*3,0));

  // TODO compare laplace to cpu version


  // test conjugate gradient
  float* x = cuda_allocate_field(N);
  float* b = cuda_allocate_field(N);
  fillArray<<<1,1024>>>(b,1,N);
  // fillArray<<<1,1024>>>(x,0,N);
  conjugate_gradient_gpu(b,x,L,d);
  CHECK(cudaDeviceSynchronize());
  float * xcpu = (float*)malloc(N*sizeof(float));
  cudaMemcpy(xcpu,x,N*sizeof(float),cudaMemcpyDeviceToHost);
  // for (int i = 0; i < N; i++)
  // {
  //   printf("%f ",xcpu[i]);
  // }
  cudaFree(x);
  cudaFree(b);
  free(xcpu);
}
