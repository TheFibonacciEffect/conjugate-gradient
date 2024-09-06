#include "gpu_utils.cuh"


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

