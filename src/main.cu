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

__global__ static void reduceAdd(float *g_idata, float *g_odata,
                                 unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // add as many as possible (= 2*(n/gridSize))
  float sum = 0.0;
  int i = idx;
  while (i < n)
  {
    sum += g_idata[i] + g_idata[i + blockDim.x];
    i += gridSize;
  }
  g_idata[idx] = sum;

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      g_idata[idx] += g_idata[idx + stride];
    }

    // synchronize within threadblock
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = g_idata[idx];
}

__global__ void printArray_gpu(float *d_array, int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {
    printf("Array[%d] = %d\n", idx, d_array[idx]);
  }
}

// julia wrapper functions
extern "C"
{

  __global__ void add(float *A, float *B)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    A[idx] += B[idx];
    return;
  }
  void add_jl(float *A, float *B, int blocks, int threads)
  {
    add<<<blocks, threads>>>(A, B);
  }

  void laplace_gpu_jl(float *ddf, float *u, float dx, int d,
                                   int L, int N, unsigned int index_mode,unsigned int blocks, unsigned int threads)
  {
    laplace_gpu<<<blocks,threads>>>(ddf,u,d,L,N,index_mode);
    CHECK(cudaDeviceSynchronize());
  }
}
