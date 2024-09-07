#include "conjugate-gradient_gpu.cuh"
#include "conjugate-gradient_cpu.h"

/**
 * @brief Fills an array with a specified value.
 *
 * This CUDA kernel function fills an array with a specified value. It uses the
 * global thread index to calculate the index of each element in the array. If
 * the calculated index is within the bounds of the array, the specified value
 * is assigned to that element.
 *
 * @param arr Pointer to the array to be filled.
 * @param value The value to fill the array with.
 * @param size The size of the array.
 */
__global__ void fillArray(float* arr, float value, int size) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (idx < size) {
        arr[idx] = value;
    }
}
/**
 * @brief Allocates a managed CUDA memory for a float field of size N.
 *
 * @param N The size of the field.
 * @return A pointer to the allocated memory.
 */


__host__ float *cuda_allocate_field(int N)
{
  float *field;
  CHECK(cudaMallocManaged(&field, (N + 1) * sizeof(float)));
  fillArray<<<1,N>>>(field,0,N);
  field[N] = 0;
  cudaDeviceSynchronize();
  return field;
}

/**
 * @brief Converts an index to coordinates in a multi-dimensional grid.
 *
 * This function takes an index, the size of the grid (L), and the number of
 * dimensions (d), and converts the index to its corresponding coordinates in
 * the grid.
 *
 * @param cords The array to store the resulting coordinates.
 * @param index The index to convert to coordinates.
 * @param L The size of the grid.
 * @param d The number of dimensions.
 * @return The array containing the resulting coordinates.
 */
__device__ int *index_to_cords_cu(int *cords, int index, int L, int d) {
  assert(index < pow(L, d) && index >= 0);
  for (int i = 0; i < d; i++)
  {
    cords[i] = index % L;
    index /= L;
  }
  return cords;
}

/**
 * @brief Calculates the index in a flattened array given the coordinates in a
 * multi-dimensional array.
 *
 * This function takes in the coordinates of a point in a multi-dimensional
 * array and returns the corresponding index in a flattened array.
 *
 * @param cords The array of coordinates.
 * @param L The size of each dimension in the multi-dimensional array.
 * @param d The number of dimensions in the array.
 * @param N The total number of elements in the flattened array.
 * @return The index in the flattened array.
 */
__device__ int get_index_gpu(int *cords, int L, int d, int N) {
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

/**
 * @brief Converts a linear index to its corresponding coordinates in a
 * multi-dimensional grid.
 *
 * This function takes a linear index, the size of the grid (L), and the number
 * of dimensions (d), and returns the coordinates of the index in the grid.
 *
 * @param index The linear index to convert.
 * @param L The size of the grid.
 * @param d The number of dimensions.
 * @return The coordinates of the index in the grid.
 */
static inline __device__ __host__ int index_to_cords(int index, int L, int d) {
  for (int i = 0; i < d; i++) {
    index /= L;
  }
  return index % L;
}

/**
 * Calculates the index of a neighboring element in a GPU grid.
 *
 * @param ind The current index.
 * @param direction The direction of the neighbor (0 to d-1).
 * @param amount The amount to move in the specified direction.
 * @param L The size of the grid in each dimension.
 * @param d The number of dimensions.
 * @param N The total number of elements in the grid.
 * @param index_mode The index mode.
 * @return The index of the neighboring element.
 */
extern "C" __host__ __device__ int neighbour_index_gpu(int ind, int direction,
                                                       int amount, int L, int d,
                                                       int N, int index_mode) {
  // should be consistant with cpu code
  int cord = index_to_cords(ind, L, direction);
  cord += amount;
  if (/*cord > L || cord < 0 ||*/ cord == -1 || cord == L)
    return N;

  // if on boundary => return 0 through special index
  assert(amount == 1 || amount == -1 || amount == 0);
  int n=1;
  for (int i=0; i<direction; i++)
  {
      n *= L;
  }
  ind += amount*n;
  
  return ind;
}

/**
 * @brief Computes the Laplace operator on a GPU for a given input array.
 *
 * This function calculates the Laplace operator on a GPU for a given input
 * array `u`. The Laplace operator is computed using the finite difference
 * method. The result is stored in the output array `ddf`.
 *
 * @param ddf - Output array to store the computed Laplace operator.
 * @param u - Input array on which the Laplace operator is computed.
 * @param d - Dimension of the input array.
 * @param L - Size of each dimension of the input array.
 * @param N - Total number of elements in the input array.
 * @param index_mode - Indexing mode for accessing neighbouring elements.
 */
__global__ void laplace_gpu(float *ddf, float *u, int d, int L, int N,
                            unsigned int index_mode) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < N)
  {
    float laplace_value = 0;
    for (int i = 0; i < d; i++)
    {
      laplace_value += - u[neighbour_index_gpu(ind, i, 1, L, d, N, index_mode)]
                       + 2 * u[neighbour_index_gpu(ind, i, 0, L, d, N, index_mode)]
                       - u[neighbour_index_gpu(ind, i, -1, L, d, N, index_mode)];
    }
    // the discrete version is defined without dx
    ddf[ind] = laplace_value; 
  }
}
/**
 * @brief Performs a complete reduction operation on the input arrays `v` and
 * `w`, and stores the result in the global memory array `g_odata`.
 *
 * This kernel function is designed to be executed on a CUDA device.
 *
 * @param v         Pointer to the input array `v`.
 * @param w         Pointer to the input array `w`.
 * @param g_odata   Pointer to the output array `g_odata`.
 * @param n         The size of the input arrays `v` and `w`.
 * @param nthreads  The number of threads per block.
 */

__global__ void reduceMulAddComplete(float *v, float *w, float *g_odata,
                                            unsigned int n,const unsigned int nthreads)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // shared memory is per block
  extern __shared__ float tmp[]; // shared memory can be given as 3rd argument to allocate it dynamicially

  // unroll as many as possible
  float sum = 0.0;
  int i = idx;
  while (i < n)
  {
    sum += v[i]* w[i] + v[i + blockDim.x]* w[i + blockDim.x];
    i += gridSize;
  }
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

/**
 * Calculates the inner product of two arrays on the GPU.
 *
 * @param v Pointer to the first array.
 * @param w Pointer to the second array.
 * @param N The size of the arrays.
 * @return The inner product of the two arrays.
 */
extern "C" float inner_product_gpu(float *v, float *w, unsigned int N) {
  float *bs, r;
  const int nthreads = 1;
  int nblocks = N/nthreads;

  // bs is only size 1 not size nblocks, this is because of the definition of atomic add
  CHECK(cudaMalloc((void **)&bs, sizeof(float))); 

  reduceMulAddComplete<<<nblocks, nthreads, nthreads*sizeof(float)>>>(v, w, bs, N, nthreads);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(&r, bs, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(bs));

  return r;
}

/**
 * @brief Calculate the norm of a vector.
 *
 * This function calculates the norm of a vector using the formula:
 * norm(v) = sqrt(inner_product_gpu(v, v, N))
 *
 * @param v The input vector.
 * @param N The size of the vector.
 * @return The norm of the vector.
 */
__host__ float norm(float *v, int N) {
  return sqrt(inner_product_gpu(v, v, N));
}

/**
 * @brief Function to perform tests.
 *
 * This function performs tests on the `index_to_cords` function.
 * It asserts the correctness of the function by comparing the returned values
 * with expected values.
 *
 * @note This function assumes that the `index_to_cords` function is implemented
 * correctly.
 *
 * @return void
 */
void tests() {
  // index
  assert(index_to_cords(4,3,1) == 1);
  assert(index_to_cords(2,3,1) == 0);
  assert(index_to_cords(2,3,0) == 2);
}

// A = b*B
__global__ void muladd(float* A, float b, float* B, int N)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= N) return;
  A[ind] = A[ind] + b*B[ind];
}

// A = C + b*B
__global__ void muladd3(float* A, float * C, float b, float* B, int N)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= N) return;
  A[ind] = C[ind] + b*B[ind];
}

/**
 * @brief Solves the laplacian using the Conjugate Gradient method on the GPU.
 *
 * This function solves a linear system of equations Ax = b using the Conjugate
 * Gradient method on the GPU, where A is the discrete laplacian. The system is
 * defined by the input parameters:
 * - `b`: Pointer to the right-hand side vector of length N.
 * - `x`: Pointer to the solution vector of length N.
 * - `L`: The size of the grid in each dimension.
 * - `d`: The number of dimensions.
 *
 * @param b Pointer to the right-hand side vector.
 * @param x Pointer to the solution vector.
 * @param L The size of the grid in each dimension.
 * @param d The number of dimensions.
 * @return The residue of the solution.
 */
extern "C" float conjugate_gradient_gpu(float *b, float *x, int L, int d) {
  int nthreads = 1;
  int N = pow(L, d);
  assert(N > nthreads);
  int nblocks = N/nthreads;
  float reltol = 1e-6*norm(b, N);
  int i = 0;
  float *r = cuda_allocate_field(N);
  muladd<<<nblocks,nthreads>>>(r,1,b,N); //assume r=0
  cudaDeviceSynchronize();
  float *Ap = cuda_allocate_field(N);
  float *p = cuda_allocate_field(N);
  // p = r
  muladd<<<nblocks,nthreads>>>(p,1,r,N); //assume p=0
  cudaDeviceSynchronize();
  float rr = inner_product_gpu(r, r, N);
  float rr_new = rr;
  float alpha = NAN;
  float residue = norm(r,N);
  float beta = 0; 
  while (residue > reltol)
  {

    laplace_gpu<<<nblocks, nthreads>>>(Ap, p, d, L, N, 0);
    CHECK(cudaDeviceSynchronize());

    alpha = rr / inner_product_gpu(p, Ap, N);
    muladd<<<nblocks, nthreads>>>(x, alpha, p, N);
    muladd<<<nblocks, nthreads>>>(r, -alpha, Ap, N);
    CHECK(cudaDeviceSynchronize());
    rr_new = inner_product_gpu(r, r, N);
    beta = rr_new / rr;

    cudaDeviceSynchronize();
    muladd3<<<nblocks, nthreads>>>(p,r, beta, p, N);
    cudaDeviceSynchronize();
    CHECK(cudaDeviceSynchronize());
    residue = sqrt(rr);
    rr = rr_new;
    i++;
  }
  cudaFree(r);
  cudaFree(Ap);
  cudaFree(p);
  CHECK(cudaDeviceSynchronize());
  return residue;
}


