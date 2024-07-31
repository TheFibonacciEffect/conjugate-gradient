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

__device__ int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
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