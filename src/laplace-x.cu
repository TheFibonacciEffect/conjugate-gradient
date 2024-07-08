#define TYPE float
#define FUNCTION(NAME) NAME##_f
#include "common.h"

static __device__ int *index_to_cords_cu(int *cords, int index, int L, int d) {
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++) {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}

// this is a type agnostic version of the laplace operator and associated
// functions on the gpu
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

__global__ void FUNCTION(apply_function_gpu)(TYPE *result, TYPE (*f)(int),
                                             int N, int L, int d) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < N) {
    int cords[dmax];
    index_to_cords_cu(cords, ind, L, d);
    result[ind] = f(cords[0]);
  }
}