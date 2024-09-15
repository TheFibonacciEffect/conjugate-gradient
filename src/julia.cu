#include "conjugate-gradient_gpu.cuh"
#include <chrono>
#include<unistd.h>

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// julia wrapper functions
extern "C"
{
  // void strong_scaling()
  // problem size stays the same
  float strong_scaling(int nblocks, int threads_per_block, int N, int L, int d)
  {
    int nthreads = threads_per_block;
    // allocate an array
    float *x = cuda_allocate_field(N);
    float *b = cuda_allocate_field(N);
    // fill it with random data
    random_array(x, L, d, N);
    auto t0 = timeSinceEpochMillisec();
    laplace_gpu<<<nblocks, nthreads>>>(b, x, d, L, N, 1);
    cudaDeviceSynchronize();
    auto t1 = timeSinceEpochMillisec();
    // printf("%d\n", t1-t0);
    return t1-t0;
  }

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
                      int L, int N, unsigned int index_mode, unsigned int blocks, unsigned int threads)
  {
    laplace_gpu<<<blocks, threads>>>(ddf, u, d, L, N, index_mode);
    CHECK(cudaDeviceSynchronize());
  }
}
