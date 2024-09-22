#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests for CPU
#include "conjugate-gradient_cpu.h"

// TESTS
float square(float x, float y) { return (x * x + y * y) / 4; }

bool every_a(double *x, double *y, int n)
{
  double tol = 1e-3;
  for (int i = 0; i < n; i++)
  {
    if (!(fabs(x[i] - y[i]) < tol)) // floating point comparison also accounting
                                    // for NaNs that always return false
    {
      return false;
    }
  }
  return true;
}

bool every_a(float *x, float *y, int n)
{
  printf("calling every a");
  float tol = 1e-3;
  for (int i = 0; i < n; i++)
  {
    if (!(fabs(x[i] - y[i]) < tol)) // floating point comparison also accounting
                                    // for NaNs that always return false
    {
      return false;
    }
  }
  return true;
}


bool test_cg(int L, int d, int N)
{
  printf("testing cg\n");
  // allocate an array
  double dx = 2.0 / (L - 1);
  double *x = allocate_field(N);
  double *b = allocate_field(N);
  // fill it with random data
  random_array(x, L, d, N);
  // calculate b
  b = minus_laplace(b, x, d, L, N);
  // initial guess
  double *x0 = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  conjugate_gradient(b, x0, L, d);
  // compare with x
  bool passed = false;
  if (every_a(x, x0, N))
  {
    passed = true;
  }
  else
  {
    passed = false;
  }
  free(x);
  free(b);
  free(x0);
  printf("result: %d\n", passed);
  return passed;
}

bool test_preconditioner(int L, int d, int N)
{
  // allocate an array
  double dx = 2.0 / (L - 1);
  double *x = allocate_field(N);
  double *b = allocate_field(N);
  // fill it with random data
  random_array(x, L, d, N);
  // calculate b
  b = minus_laplace(b, x, d, L, N);
  // initial guess
  double *x0 = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  preconditioner(b, x0, L, d, 1e-6);
  // compare with x
  bool passed = false;
  if (every_a(x, x0, N))
  {
    passed = true;
  }
  else
  {
    passed = false;
  }
  free(x);
  free(b);
  free(x0);
  return passed;
}

bool test_preconditioned_cg(int L, int d, int N)
{
  // allocate an array
  double dx = 2.0 / (L - 1);
  double *x = allocate_field(N);
  double *b = allocate_field(N);
  // fill it with random data
  random_array(x, L, d, N);
  // calculate b
  b = minus_laplace(b, x, d, L, N);
  // initial guess
  double *x0 = allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  x0 = preconditioned_cg(b, x0, L, d);
  // compare with x
  bool passed = false;
  if (every_a(x, x0, N))
  {
    passed = true;
  }
  else
  {
    passed = false;
  }
  free(x);
  free(b);
  free(x0);
  return passed;
}

bool is_boundary(int *cords, int L, int d)
{
  for (int i = 0; i < d; i++)
  {
    if (cords[i] == 0 || cords[i] == L - 1)
    {
      return true;
    }
  }
  return false;
}

bool test_laplace()
{
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int i;
  double dx = 2.0 / (L - 1);
  double x, y;
  double *u = allocate_field(N);
  u[N] = 0;
  for (i = 0; i < N; i++)
  {
    int cords[2];
    index_to_cords(cords, i, L, d);
    int nx = cords[0];
    int ny = cords[1];
    x = -1 + nx * dx;
    y = -1 + ny * dx;
    u[i] = square(x, y);
  }
  double *ddf = allocate_field(N);
  ddf = minus_laplace(ddf, u, d, L, N);

  for (int i = 0; i < N; i++)
  {
    int cords[2];
    index_to_cords(cords, i, L, d);
    int nx = cords[0];
    int ny = cords[1];
    x = -1 + nx * dx;
    y = -1 + ny * dx;
    if (!is_boundary(cords, L, d) &&
        !(ddf[i] - 1 > 1e-3 || ddf[i] - 1 < -1e-3))
    {
      printf("Test failed\n");
      printf("x: %f, y: %f, ddf: %f\n", x, y, ddf[i]);
      return false;
    }
  }
  free(u);
  free(ddf);
  return true;
}

bool run_test_gc_cpu()
{
  int L = 5;
  int d = 3;
  int N = pow(L, d);
  return test_cg(L, d, N);
}

bool test_inner_product()
{
  double x[3] = {1, 2, 3};
  double y[3] = {4, 5, 6};
  double result = inner_product(x, y, 3);
  if (result - 32 > 1e-3 || result - 32 < -1e-3)
  {
    return false;
  }
  return true;
}

bool test_norm()
{
  double x[3] = {1, 2, 3};
  double result = norm(x, 3);
  if (result - sqrt(14) > 1e-3 || result - sqrt(14) < -1e-3)
  {
    return false;
  }
  return true;
}

bool test_getindex()
{
  int cords[2] = {1, 2};
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int result = get_index(cords, L, d, N);
  if (result != 11)
  {
    return false;
  }
  return true;
}

bool test_getindex_edge()
{
  int cords[2] = {5, 5}; // outside the array
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int result = get_index(cords, L, d, N);
  if (result != N)
  {
    return false;
  }
  return true;
}

bool test_getindex_edge2()
{
  int cords[2] = {-1, 0}; // outside the array
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int result = get_index(cords, L, d, N);
  if (result != N)
  {
    return false;
  }
  return true;
}

bool test_neighbour_index()
{
  int cords[2] = {0, 2};
  int direction = 1;
  int amount = -1;
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int result = neighbour_index(cords, direction, amount, L, d, N);
  if (result != 5)
  {
    return false;
  }
  return true;
}

bool test_neighbour_index2()
{
  int cords[2] = {1, 2};
  int direction = 1;
  int amount = -1;
  int L = 5;
  int d = 2;
  int N = pow(L, d);
  int result = neighbour_index(cords, direction, amount, L, d, N);
  if (result != 6)
  {
    return false;
  }
  return true;
}

extern int run_tests_cpu()
{
  printf("running tests");
  assert(test_laplace());
  assert(test_inner_product());
  assert(test_norm());
  assert(test_getindex());
  assert(test_getindex_edge2());
  assert(test_getindex_edge());
  assert(test_neighbour_index());
  assert(run_test_gc_cpu());
  assert(test_preconditioner(50, 2, 2500));
  printf("test preconditioned cg\n");
  assert(test_preconditioned_cg(50, 2, 2500));
  printf("Tests Passed!\n");
  return 0;
}

// Tests for GPU
#include "conjugate-gradient_gpu.cuh"

__global__ void squareKernel(float *d_array, int n, float step)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    float x = -M_PI + idx * step;
    d_array[idx] = x * x;
  }
}

__global__ void sinKernel(float *d_array, int n, float step)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    float x = -M_PI + idx * step;
    d_array[idx] = sin(x);
  }
}

static void test_inner_product_gpu()
{
  int N = 1000;
  float *x = cuda_allocate_field(N);
  fillArray<<<1, 1024>>>(x, 1, N);
  float r = inner_product_gpu(x, x, N);
  printf("x*x =%f\n", r);
  assert(r == N);
  cudaFree(x);
}

// CUDA kernel to perform element-wise division
__global__ void div(float *d_array1, float *d_array2, float *d_result, int n)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    if (d_array2[idx] != 0)
    { // Check to avoid division by zero
      d_result[idx] = d_array1[idx] / d_array2[idx];
    }
    else
    {
      d_result[idx] = NAN;
    }
  }
}

static void test_laplace_square()
{
  int N = 1000;
  float step = (2 * M_PI) / (N - 1);
  float *ddf = cuda_allocate_field(N);
  float *u = cuda_allocate_field(N);
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  int L = N;
  int d = 1;
  unsigned int index_mode = 0;

  squareKernel<<<blocksPerGrid, threadsPerBlock>>>(u, N, step);
  CHECK(cudaDeviceSynchronize());
  laplace_gpu<<<blocksPerGrid, threadsPerBlock>>>(ddf, u, d, L, N, index_mode);
  CHECK(cudaDeviceSynchronize());
  float *ddf_c = (float *)malloc(N * sizeof(float));
  cudaMemcpy(ddf_c, ddf, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 1; i < N - 1; i++) // all except boundary
  {
    // printf("%f ",ddf_c[i] -  ddf_c[N/2]); // all the same value
    assert(abs(ddf_c[i] - ddf_c[N / 2]) < 1e-3);
  }
  cudaFree(ddf);
  cudaFree(u);
}

static void test_laplace_sin()
{
  int N = 1000;
  float step = (2 * M_PI) / (N - 1);
  float *ddf = cuda_allocate_field(N);
  float *u = cuda_allocate_field(N);
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  int L = N;
  int d = 1;
  unsigned int index_mode = 0;

  sinKernel<<<blocksPerGrid, threadsPerBlock>>>(u, N, step);
  CHECK(cudaDeviceSynchronize());
  laplace_gpu<<<blocksPerGrid, threadsPerBlock>>>(ddf, u, d, L, N, index_mode);
  CHECK(cudaDeviceSynchronize());
  div<<<blocksPerGrid, threadsPerBlock>>>(ddf, u, ddf, N);
  CHECK(cudaDeviceSynchronize());
  float *ddf_c = (float *)malloc(N * sizeof(float));
  cudaMemcpy(ddf_c, ddf, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 1; i < N - 1; i++) // all except boundary
  {
    // printf("%f ",ddf_c[i] -  ddf_c[N/2]); // all the same value
    assert(abs(ddf_c[i] - ddf_c[N / 2]) < 1e-3);
  }
  cudaFree(ddf);
  cudaFree(u);
}

int ipow(int N, int d)
{
  int k=1;
  for (int i = 0; i < d; i++)
  {
    k = k*N;
  }
  return N;
}

static void test_laplace_large()
{
  // created this test case, because when calling from julia with large arrays sometimes there is an illigal memory access
  // this is to make sure that this is not a problem with the C code.
  int L = 100;
  int d = 5;
  int N = ipow(L,d);
  printf("N: %d\n", N);
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // assert(2*N < 2000000000); // 8GB / 32 bit
  float *ddf = cuda_allocate_field(N);
  float *u = cuda_allocate_field(N);
  laplace_gpu<<<blocksPerGrid, threadsPerBlock>>>(ddf, u, d, L, N, 0);
  CHECK(cudaDeviceSynchronize());
  div<<<blocksPerGrid, threadsPerBlock>>>(ddf, u, ddf, N);
  CHECK(cudaDeviceSynchronize());
  float *ddf_c = (float *)malloc(N * sizeof(float));
  cudaMemcpy(ddf_c, ddf, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 1; i < N - 1; i++) // all except boundary
  {
    // printf("%f ",ddf_c[i] -  ddf_c[N/2]); // all the same value
    assert(abs(ddf_c[i] - ddf_c[N / 2]) < 1e-3);
  }
  cudaFree(ddf);
  cudaFree(u);
}

bool test_cg_gpu(int L, int d, int N)
{
  int nblocks = N;
  int nthreads = 1;
  printf("testing cg\n");
  // allocate an array
  float *x = cuda_allocate_field(N);
  float *b = cuda_allocate_field(N);
  // fill it with random data
  random_array(x, L, d, N);
  // calculate b
  laplace_gpu<<<nblocks, nthreads>>>(b, x, d, L, N, 1);
  // initial guess
  float *x0 = cuda_allocate_field(N);
  for (int i = 0; i < N; i++)
  {
    x0[i] = 0;
  }
  // apply conjugate gradient to calculate x
  conjugate_gradient_gpu(b, x0, L, d);
  // compare with x
  bool passed = false;
  if (every_a(x, x0, N))
  {
    passed = true;
  }
  else
  {
    passed = false;
  }
  cudaFree(x);
  cudaFree(b);
  cudaFree(x0);
  printf("result: %d\n", passed);
  return passed;
}

// tests for interleaved indexing
#include "interleave.cuh"

int main(int argc, char const *argv[])
{
  // CPU Tests
  printf("running cpu tests\n");
  run_tests_cpu();
  // GPU tests
  printf("running gpu tests\n");
  test_inner_product_gpu();
  test_laplace_sin();
  test_laplace_square();
  test_laplace_large();
  printf("testing inteleaved indexing \n");
  assert(test_cg_gpu(10, 2, 100));

  tests_interleaved_index();

  printf("tests passed!\n");

  return 0;
}
