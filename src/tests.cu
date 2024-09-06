#include "gpu_utils.cuh"

static void test_laplace_sin()
{
  int N = 1000;
  float step = (2 * M_PI) / (N - 1);
  float * ddf = cuda_allocate_field(N);
  float * u = cuda_allocate_field(N);
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  int L = N;
  int d = 1;
  unsigned int index_mode = 0;

  sinKernel<<<blocksPerGrid, threadsPerBlock>>>(u, N, step);
  CHECK(cudaDeviceSynchronize());
  laplace_gpu<<<blocksPerGrid,threadsPerBlock>>>(ddf,u,d,L,N,index_mode);
  CHECK(cudaDeviceSynchronize());
  div<<<blocksPerGrid,threadsPerBlock>>>(ddf,u,ddf,N);
  CHECK(cudaDeviceSynchronize());
  float * ddf_c = (float*)malloc(N*sizeof(float));
  cudaMemcpy(ddf_c,ddf,N*sizeof(float),cudaMemcpyDeviceToHost);
  for (int i = 1; i < N-1; i++) // all except boundary
  {
    // printf("%f ",ddf_c[i] -  ddf_c[N/2]); // all the same value
    assert(abs(ddf_c[i] - ddf_c[N/2]) < 1e-3);
  }
  cudaFree(ddf);
  cudaFree(u);
}


int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}

