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
#include "preconditioner_gpu.h"


int main()
{
  // printf("%d\n",index_to_cords(10,3,2)); // 3x3x3 cube 
  printf("%d\n",neighbour_index_gpu(0,1,1,3,3,3*3*3,0)); 
  printf("%d\n",neighbour_index_gpu(3,1,1,3,3,3*3*3,0));
  printf("%d\n",neighbour_index_gpu(6,1,1,3,3,3*3*3,0));

  // printf("%d\n",neighbour_index_gpu(13,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,1,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(16,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,2,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(19,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,3,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(22,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,4,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(25,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,5,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(10,1,-1,3,3,3*3*3,0)); // 3x3x3 cube (1,0,0) - (0,-1,0) => out of bounds
  
  test_inner_product();
  test_laplace_square();
  test_laplace_sin();
  // TODO compare laplace to cpu version


  // test conjugate gradient
  int N = 100;
  int L = N;
  int d = 1;
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
