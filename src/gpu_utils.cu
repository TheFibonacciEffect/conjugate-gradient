#include "gpu_utils.cuh"

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

static inline __device__ __host__ int index_to_cords(int index, int L, int d) {
  for (int i = 0; i < d; i++) {
    index /= L;
  }
  return index % L;
}

extern "C" __host__ __device__ int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
                    int N, int index_mode)
{
  // if on boundary => return 0 through special index
  // TODO Is there hardware side bounds checking? (See intel MPX)
  for (int c = 0; c < d; c++)
  {
    // should be consistant with cpu code
    int cord =  index_to_cords(ind,L,c);
    printf("%d\n",cord);
    if (cord > L || cord < 0 || cord == -1 || cord == L) return N;
  }
  
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
  A[ind] = b*B[ind];
}

extern "C" float conjugate_gradient_gpu(float * b, float * x , int L, int d)
{
  int nthreads = 265;
  int N = pow(L, d);
  assert(N > nthreads);
  int nblocks = N/nthreads +1;
  float reltol = 1e-6*norm(b, N);
  int i = 0;
  float rr = 0,rr_new = 0,alpha = 0,residue = 0,beta = 0;
  float *r = cuda_allocate_field(N);
  float *Ap = cuda_allocate_field(N);
  float *p = cuda_allocate_field(N);
  printf("%f > %f \n" , residue, reltol);
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
  cudaFree(r);
  cudaFree(Ap);
  cudaFree(p);
  return residue;
}

__global__ void fillArray(float* arr, float value, int size) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (idx < size) {
        arr[idx] = value;
    }
}

void test_inner_product()
{
  // TODO test fails
  int N = 1000;
  float* x = cuda_allocate_field(N);
  float r = inner_product_gpu(x,x,N);
  printf("x*x =%f\n",r);
  cudaFree(x);
}

int main()
{
  // printf("%d\n",index_to_cords(10,3,2)); // 3x3x3 cube 
  printf("%d\n",neighbour_index_gpu(0,1,1,3,3,3*3*3,0)); // 0 0 0 
  printf("%d\n",neighbour_index_gpu(3,1,1,3,3,3*3*3,0)); // 0 1 0
  printf("%d\n",neighbour_index_gpu(6,1,1,3,3,3*3*3,0)); // 0 2 0
  printf("%d\n",neighbour_index_gpu(9,1,1,3,3,3*3*3,0)); // 0 0 1 => WRONG!

  // printf("%d\n",neighbour_index_gpu(13,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,1,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(16,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,2,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(19,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,3,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(22,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,4,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(25,1,1,3,3,3*3*3,0)); // 3x3x3 cube (1,5,0) + (0,1,0)
  // printf("%d\n",neighbour_index_gpu(10,1,-1,3,3,3*3*3,0)); // 3x3x3 cube (1,0,0) - (0,-1,0) => out of bounds
  
  test_inner_product();

  // test conjugate gradient
  int N = 1000;
  int L = N;
  int d = 1;
  float* x = cuda_allocate_field(N);
  float* b = cuda_allocate_field(N);
  fillArray<<<1,1024>>>(b,1,N);
  conjugate_gradient_gpu(b,x,L,d);
  CHECK(cudaDeviceSynchronize());
  float * xcpu = (float*)malloc(N*sizeof(float));
  cudaMemcpy(xcpu,x,N*sizeof(float),cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
  {
    printf("%f\n",xcpu[i]);
  }
  cudaFree(x);
  cudaFree(b);
  free(xcpu);
  
}
