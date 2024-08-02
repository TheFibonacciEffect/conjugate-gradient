#include <stdio.h>
#include "common.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>

__host__ float *cuda_allocate_field(int N)
{
  float *field;
  CHECK(cudaMallocManaged(&field, (N + 1) * sizeof(float)));
  field[N] = 0;
  return field;
}

__device__ int *index_to_cords_cu(int *cords, int index, int L, int d)
{
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++)
  {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}

__device__ int get_index_gpu(int *cords, int L, int d, int N)
{
  int index = 0;
  for (int i = 0; i < d; i++)
  {
    index += cords[i];
    if (i < d - 1)
    {
      index *= L;
    }
  }
  return index;
}

extern "C" __host__ __device__ int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
                    int N, int index_mode)
{
  assert(amount == 1 || amount == -1 || amount == 0);
  int n=1;
  for (int i=0; i<direction; i++)
  {
      n *= L;
  }
  ind += amount*n;
  return ind;
}

/* index_mode:
0 = naiive
1 = bit mixing
2 = lookup table */
__global__ void laplace_gpu(float *ddf, float *u, int d,
                                   int L, int N, unsigned int index_mode)
{
// if (index_mode > 2) throw std::invalid_argument( "received invalid index" );
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < N)
  {
    float laplace_value = 0;
    for (int i = 0; i < d; i++)
    {
      laplace_value += -u[neighbour_index_gpu(ind, i, 1, L, d, N, index_mode)] +
                       2 * u[neighbour_index_gpu(ind, i, 0, L, d, N, index_mode)] -
                       u[neighbour_index_gpu(ind, i, -1, L, d, N, index_mode)];
    }
    // the discrete version is defined without dx
    // ddf[ind] = laplace_value / pow(dx, d);
  }
}

__global__ void reduceMulAddComplete(float *v, float *w, float *g_odata,
                                            unsigned int n,const unsigned int nthreads)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  extern __shared__ float tmp[]; // shared memory can be given as 3rd argument to allocate it dynamicially

  // unroll as many as possible
  float sum = 0.0;
  int i = idx;
  while (i < n)
  {
    sum += v[i]* w[i] + v[i + blockDim.x]* w[i + blockDim.x];
    i += gridSize;
  }
  // g_idata[idx] = sum;
  tmp[tid] = sum;

  __syncthreads();

  // in-place reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      tmp[tid] += tmp[tid + stride];
    }

    // synchronize within threadblock
    __syncthreads();
  }

  // atomicAdd result of all blocks to global mem
  if (tid == 0)
    atomicAdd(g_odata, tmp[0]);
}

extern "C" float inner_product_gpu(float *v, float *w, unsigned int N)
{
  float *bs, r;
  const int nthreads = 265;
  int nblocks = N/nthreads +1;

  // bs is only size 1 not size nblocks?
  CHECK(cudaMalloc((void **)&bs, sizeof(float))); 

  reduceMulAddComplete<<<nblocks, nthreads, nthreads*sizeof(float)>>>(v, w, bs, N, nthreads);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(&r, bs, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(bs));

  return r;
}

__host__ float norm(float *v, int N)
{
  return sqrt(inner_product_gpu(v, v, N));
}

// TODO still work in progress
// initial guess is 0
/* extern "C" float conjugate_gradient_gpu(float * b, float * x , int L, int d)
{
  int nthreads = 265;
  int N = pow(L, d);
  assert(N > nthreads);
  int nblocks = N/nthreads +1;
  float *r = cuda_allocate_field(N);
  float residue = 0;
  float reltol = 1e-6*norm(b, N);
  while (residue > reltol)
  {
    i++;
    laplace_gpu<<<nblocks, nthreads>>>(Ap, p, d, L, N, 0);
    CHECK(cudaDeviceSynchronize());
    alpha = rr / inner_product_gpu(p, Ap, N);
    muladd<<<nblocks, nthreads>>>(x, alpha, p, N);
    muladd<<<nblocks, nthreads>>>(r, -alpha, Ap, N);
    CHECK(cudaDeviceSynchronize());
    rr_new = inner_product_gpu(r, r, N);
    beta = rr_new / rr;
    muladd<<<nblocks, nthreads>>>(p, beta, p, N);
    CHECK(cudaDeviceSynchronize());
    rr = rr_new;
    printf("residue: %f at iteration: %i\n", residue, i);
  }
  return residue;
} */
