#include <stdio.h>
// #include <math.h>
#include "common.h"
#include "interleave.cuh"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// #define L 5  // Lattice size
// #define N (int)pow(L,d) // Number of lattice points

#include "conjugate-gradient_cpu.h"

#include "preconditioner_gpu.h"

__device__ int get_index_gpu(int *cords, int L, int d, int N) {
  int index = 0;
  for (int i = 0; i < d; i++) {
    index += cords[i];
    if (i < d - 1) {
      index *= L;
    }
  }
  return index;
}

__device__ int neighbour_index_gpu(int *cords, int direction, int amount, int L,
                                   int d, int N) {
  cords[direction] += amount;
  int ind = get_index_gpu(cords, L, d, N);
  cords[direction] -= amount;
  return ind;
}

__device__ int *index_to_cords_cu(int *cords, int index, int L, int d) {
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++) {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}

__global__ void ones(float *x) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  x[ind] = 1;
}

#define TYPE float
#define FUNCTION(NAME) NAME##_f
#define dmax 5

// __global__ void FUNCTION(apply_function_gpu)(TYPE * result, int N, int L, int
// d) {
//     int ind = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int i=ind; i < N; i+=blockDim.x) {
//         int cords[dmax];
//         index_to_cords_cu(cords, ind,  L, d);
//         result[ind] = f(cords[0]);
//     }
// }

__host__ TYPE *FUNCTION(cuda_allocate_field)(int N) {
  TYPE *field;
  CHECK(cudaMallocManaged(&field, (N + 1) * sizeof(TYPE)));
  field[N] = 0;
  return field;
}

__global__ void FUNCTION(minus_laplace_gpu_)(TYPE *ddf, TYPE *u, TYPE dx, int d,
                                             int L, int N) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < N) {
    int cords[dmax];
    index_to_cords_cu(cords, ind, L, d);
    TYPE laplace_value = 0;
    for (int i = 0; i < d; i++) {
      laplace_value += -u[neighbour_index_gpu(cords, i, 1, L, d, N)] +
                       2 * u[neighbour_index_gpu(cords, i, 0, L, d, N)] -
                       u[neighbour_index_gpu(cords, i, -1, L, d, N)];
    }
    ddf[ind] = laplace_value / pow(dx, d);
  }
}

// __global__ void sum(int* g_idata, int* g_odata)
// {
//    __shared__ int sdata[NTHREADS]; //why do I not need to reserve this?

//    // each itteration loads one element from global to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//    sdata[tid] = g_idata[i];
//    __syncthreads();

//    // do reduction in shared memory
//    // why cant I do it in the normal memory (whatever that is)?
//    for (unsigned int s = 1; s < blockDim.x; s*=2)
//    {
//       if (tid % (2*s) == 0)
//       {
//          sdata[tid] += sdata[tid +s];
//       }
//       __syncthreads();
//    }
//    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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

// TODO write and test this
__global__ void inner_product_1(float *a, const float *b, float *tmp, const int n,
                                int arretmetic_complexity) {
  assert(blockDim.x * gridDim.x > n);
  extern __shared__ float array[];
  unsigned int tid = threadIdx.x;
  unsigned int i =
      blockIdx.x * blockDim.x + arretmetic_complexity * threadIdx.x;
  if (i > n)
    return;
  float result = 0;
  for (int j = 0; j < arretmetic_complexity && i + j < n; j++) {
    result += a[i + j] * b[i + j];
  }
  a[i] = result;
  __syncthreads();

  // something like this
  /*     // in-place reduction in global memory
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
  if (tid == 0) g_tmp[blockIdx.x] = g_idata[idx]; */

  // TODO
  if (tid == 0)
  {
      for (int j = 0; j < n; j += arretmetic_complexity)
      {
        tmp[blockIdx.x] += a[j];
      }
  }
  return;
}

__global__ void sum(float *tmp, int N) {
  // // do the rest of the reduction
  assert(gridDim.x == 1);
  // todo

}

float inner_product(float *a, float *b, int N) {

  constexpr int nthread = 224;
  int nblocks = N/nthread;
  assert(N % (nthread) == 0);
  // // tmp array
  float * tmp;
  CHECK(cudaMalloc(&tmp, nblocks*sizeof(float)));

  inner_product_1<<<N/nthread, nthread>>>(a,b,tmp,N,5);
  gpuErrchk( cudaDeviceSynchronize() );
  sum<<<1, nblocks>>>(tmp,nblocks);
  // gpuErrchk( cudaDeviceSynchronize() );
  // gpuErrchk( cudaPeekAtLastError() );
  // free(tmp);
}

__global__ void printArray_gpu(float* d_array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        printf("Array[%d] = %d\n", idx, d_array[idx]);
    }
}


__global__ void FUNCTION(print_array)(TYPE *array, int N) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  while (ind < N) {
    printf("%f ", array[ind]);
    ind += blockDim.x * gridDim.x;
  }
}

void preconditioner_gpu(float *b, float *x, int L, int d, float errtol) {
  int N = pow(L, d);
  float *r = cuda_allocate_field_f(N);
  cudaMemcpy(r, b, N * sizeof(float), cudaMemcpyHostToDevice);
  float *tmp = cuda_allocate_field_f(N); // tmp field
  cudaMemcpy(tmp, r, N * sizeof(float), cudaMemcpyDeviceToDevice);
}

int tests_interleaved_index() {
  Coords<2> c({2, 3});
  Index<2> i = from_coords(c);
  auto i2 = i.neighbour<1>(1);
  auto i3 = i.neighbour<1>(0);
  auto i4 = i.neighbour<-1>(1);
  auto i5 = i.neighbour<-1>(0);
  auto c_back = i.to_coords();
  auto c2 = i2.to_coords();
  auto c3 = i3.to_coords();
  auto c4 = i4.to_coords();
  auto c5 = i5.to_coords();
  int *p = (int *)malloc(sizeof(int) * 100);
  PtrND<int, 2> ptr{p};
  ptr[i] = 10;
  printf("%#x = %d, %d\n", i.i, c_back[0], c_back[1]);
  printf("%#x = %d, %d\n", i2.i, c2[0], c2[1]);
  printf("%#x = %d, %d\n", i3.i, c3[0], c3[1]);
  printf("%#x = %d, %d\n", i4.i, c4[0], c4[1]);
  printf("%#x = %d, %d\n", i5.i, c5[0], c5[1]);
  printf("ptr[i] = %d\n", ptr[i]);
  assert(c_back[0] == 2);
  assert(c_back[1] == 3);

  Coords<2> cover({0, 0});
  auto iover = from_coords(cover);
  auto ioverleft = iover.neighbour<-1>(0);
  auto coverleft = ioverleft.to_coords();
  printf("%#x = %d %d (outside? %d)\n", ioverleft.i, coverleft[0], coverleft[1],
         ioverleft.is_outside(16 * 16));

  return 0;
}


int main() {
  run_tests_cpu();
  int L = 5;
  int d = 4;
  int N = 32;
  float *b = cuda_allocate_field_f(N);
  float *x = cuda_allocate_field_f(N);
  ones<<<N, 256>>>(b);
  float * tmp;
  CHECK(cudaMalloc(&tmp, sizeof(float)));
  reduceAdd<<<1,N>>>(b, tmp, N);
  gpuErrchk(cudaDeviceSynchronize());
  float res = NAN;
  cudaMemcpy(&res, tmp, sizeof(float), cudaMemcpyDeviceToHost);
  printf("res = %f\n", res);
  // int threadsPerBlock = 256;
  // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // printArray_gpu<<<blocksPerGrid, threadsPerBlock>>>(tmp, N);
  
  // tests_interleaved_index();
  // TODO
  // inner_product(b,b, working_memory,N,2);
  // minus_laplace_gpu_d<d,L,N><<<N,265>>>(ddf,u,dx);
  // preconditioner_gpu(b, x, L, d, 1e-3);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

extern "C" {
  void call_main() {
    main();
  }
}

