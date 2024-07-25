#include <stdio.h>
// #include <math.h>
#include "common.h"
// #include "interleave.cuh"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "preconditioner_gpu.h"

__host__ float * cuda_allocate_field(int N) {
  float *field;
  CHECK(cudaMallocManaged(&field, (N + 1) * sizeof(float)));
  field[N] = 0;
  return field;
}

// __host__ int neighbour_index_gpu(ind, direction, quantity, L,d,N);

// __global__ void minus_laplace_gpu_(float *ddf, float *u, float dx, int d,
//                                              int L, int N) {
//   int ind = blockIdx.x * blockDim.x + threadIdx.x;
//   if (ind < N) {
//     float laplace_value = 0;
//     for (int i = 0; i < d; i++) {
//       laplace_value += -u[neighbour_index_gpu(ind, i, 1, L, d, N)] +
//                        2 * u[neighbour_index_gpu(ind, i, 0, L, d, N)] -
//                        u[neighbour_index_gpu(ind, i, -1, L, d, N)];
//     }
//     ddf[ind] = laplace_value / pow(dx, d);
//   }
// }

__global__ static void reduceAdd(float *g_idata, float *g_odata,
                                 unsigned int n) {
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // add as many as possible (= 2*(n/gridSize))
  float sum = 0.0;
  int i = idx;
  while (i < n) {
    sum += g_idata[i] + g_idata[i + blockDim.x];
    i += gridSize;
  }
  g_idata[idx] = sum;

  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      g_idata[idx] += g_idata[idx + stride];
    }

    // synchronize within threadblock
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = g_idata[idx];
}

__global__ void printArray_gpu(float* d_array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        printf("Array[%d] = %d\n", idx, d_array[idx]);
    }
}

// julia wrapper functions
extern "C" {

  __global__ void add(float* A, float* B)
  {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      A[idx] += B[idx];
      return;
  }
  void print_array_gpu_jl(float* A, int size){
    printArray_gpu<<<size,1>>>(A,size);
  }
  void add_jl(float* A, float* B, int blocks, int threads)
  {
    add<<<blocks,threads>>>(A,B);
  }
}


