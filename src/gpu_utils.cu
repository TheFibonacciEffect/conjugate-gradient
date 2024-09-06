#include "gpu_utils.cuh"
#include "conjugate-gradient_cpu.h"


__global__ void fillArray(float* arr, float value, int size) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (idx < size) {
        arr[idx] = value;
    }
}


__host__ float *cuda_allocate_field(int N)
{
  float *field;
  CHECK(cudaMallocManaged(&field, (N + 1) * sizeof(float)));
  fillArray<<<1,N>>>(field,0,N);
  field[N] = 0;
  cudaDeviceSynchronize();
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

static inline __device__ __host__ int index_to_cords(int index, int L, int d) {
  for (int i = 0; i < d; i++) {
    index /= L;
  }
  return index % L;
}

extern "C" __host__ __device__ int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
                    int N, int index_mode)
{

    // 3x3 3 => x = 0, y = 1

    // 0 1 2
    // 3 4 5
    // 6 7 8

    // for c = 0 => 0
    // for c = 1 => 1

    // should be consistant with cpu code
    int cord =  index_to_cords(ind,L,direction);
    cord += amount;
    if (/*cord > L || cord < 0 ||*/ cord == -1 || cord == L) return N;

  // if on boundary => return 0 through special index
  // TODO Is there hardware side bounds checking? (See intel MPX)
  assert(amount == 1 || amount == -1 || amount == 0);
  int n=1;
  for (int i=0; i<direction; i++)
  {
      n *= L;
  }
  ind += amount*n;
  
  return ind;
}

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
      laplace_value += - u[neighbour_index_gpu(ind, i, 1, L, d, N, index_mode)]
                       + 2 * u[neighbour_index_gpu(ind, i, 0, L, d, N, index_mode)]
                       - u[neighbour_index_gpu(ind, i, -1, L, d, N, index_mode)];
    }
    // the discrete version is defined without dx
    ddf[ind] = laplace_value; 
  }
}

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
  const int nthreads = 1024;
  int nblocks = N/nthreads +1;

  // bs is only size 1 not size nblocks, this is because of the definition of atomic add
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

extern "C" float conjugate_gradient_gpu(float * b, float * x , int L, int d)
{
  int nthreads = 1;
  int N = pow(L, d);
  assert(N > nthreads);
  int nblocks = N/nthreads +1;
  float reltol = 1e-6*norm(b, N);
  int i = 0;
  float *r = cuda_allocate_field(N);
  muladd<<<nblocks,nthreads>>>(r,1,b,N); //assume r=0
  cudaDeviceSynchronize();
  printf("r= %f\n",r[5]);
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
  printf("%f > %f \n" , residue, reltol);
  while (residue > reltol)
  {
    printf("iteration %d\n",i);

    laplace_gpu<<<nblocks, nthreads>>>(Ap, p, d, L, N, 0);
    CHECK(cudaDeviceSynchronize());


    printf("array p: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", p[i]);
    }
    printf("\n");
    
    printf("array Ap: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", Ap[i]);
    }
    printf("\n");
    
    printf("array r: ");
    for (int i = 0; i < N+1; i++)
    {
      printf("%f, ", r[i]);
    }
    printf("\n");


    printf("inner_product_gpu(p, p, N): %f\n",inner_product_gpu(p, p, N));
    printf("inner_product_gpu(p, Ap, N): %f\n",inner_product_gpu(p, Ap, N));
    alpha = rr / inner_product_gpu(p, Ap, N);
    muladd<<<nblocks, nthreads>>>(x, alpha, p, N);
    muladd<<<nblocks, nthreads>>>(r, -alpha, Ap, N);
    CHECK(cudaDeviceSynchronize());
    rr_new = inner_product_gpu(r, r, N);
    printf("rrnew : %f, alpha:  %f ", rr_new, alpha);
    beta = rr_new / rr;

    cudaDeviceSynchronize();
    muladd3<<<nblocks, nthreads>>>(p,r, beta, p, N);
    cudaDeviceSynchronize();
    // check orthoginality of p
    printf("inner_product_gpu(p_old, Ap, N) should be 0: %f\n",inner_product_gpu(p, Ap, N));

    CHECK(cudaDeviceSynchronize());
    printf("rr: %f\n",rr);
    residue = sqrt(rr); //TODO replace sqrt by square => faster
    rr = rr_new;
    printf("residue: %f at iteration: %i\n", residue, i);
    i++;
  }
  cudaFree(r);
  cudaFree(Ap);
  cudaFree(p);
  return residue;
}


