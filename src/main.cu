#include <stdio.h>
// #include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "common.h"

// #define L 5  // Lattice size
// #define N (int)pow(L,d) // Number of lattice points

#include "conjugate-gradient_cpu.h"


#include "preconditioner_gpu.h"


__device__ int get_index_gpu(int* cords, int L, int d, int N) {
    int index = 0;
    for (int i = 0; i < d; i++) {
        index += cords[i];
        if (i < d - 1) {
            index *= L;
        }
    }
    return index;
}

__device__ int neighbour_index_gpu(int* cords, int direction, int amount, int L, int d, int N) {
    cords[direction] += amount;
    int ind = get_index_gpu(cords, L,d,N);
    cords[direction] -= amount;
    return ind;
}


__device__ int* index_to_cords_cu(int*cords, int index, int L, int d) {
    assert(index < pow(L,d) && index >= 0);
    for ( int i=0; i<d; i++)
    {
        cords[i] = index % L;
        index /= L;
    }
    return cords;
}


# define TYPE float
#define FUNCTION(NAME) NAME ## _f
# define dmax 5

// __global__ void FUNCTION(apply_function_gpu)(TYPE * result, int N, int L, int d) {
//     int ind = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int i=ind; i < N; i+=blockDim.x) {
//         int cords[dmax];
//         index_to_cords_cu(cords, ind,  L, d);
//         result[ind] = f(cords[0]);
//     }
// }

__host__ TYPE* FUNCTION(cuda_allocate_field)(int N) {
    TYPE * field;
    CHECK(cudaMallocManaged(&field, (N+1)*sizeof(TYPE)));
    field[N] = 0;
    return field;
}


__global__ void FUNCTION(minus_laplace_gpu_)(TYPE * ddf, TYPE * u, TYPE dx, int d, int L, int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int cords[dmax];
        index_to_cords_cu(cords,ind, L, d);
        TYPE laplace_value = 0;
        for (int i = 0; i < d; i++)
        {
            laplace_value += -u[neighbour_index_gpu(cords,i,1, L,d,N)] + 2* u[neighbour_index_gpu(cords, i, 0, L,d,N)] - u[neighbour_index_gpu(cords, i, -1, L,d,N)];
        }
        ddf[ind] = laplace_value/pow(dx,d);
    }
}

// __global__ void sum(int* g_idata, int* g_odata)
// {
//    __shared__ int sdata[NTHREADS]; //why do I not need to reserve this?

//    // each itteration loads one element from global to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//    sdata[tid] = g_idata[i];
//    __syncthreads();

//    // do reduction in shared memory
//    // why cant I do it in the normal memory (whatever that is)?
//    for (unsigned int s = 1; s < blockDim.x; s*=2)
//    {
//       if (tid % (2*s) == 0)
//       {
//          sdata[tid] += sdata[tid +s];
//       }
//       __syncthreads();
//    }
//    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }


__global__ void inner_product(TYPE *result, TYPE *a, TYPE *b, int n, int arretmetic_complexity)
{
   assert(blockDim.x*gridDim.x > n);
   int * sdata = new int[blockDim.x];
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x + arretmetic_complexity*threadIdx.x;

   if (i > n) return;
   for (int j = 0; j < arretmetic_complexity && i+j < n; j++)
   {
      result[tid] += a[i+j] * b[i+j];
   }

   return;
}


__global__ void FUNCTION(print_array)(TYPE * array, int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    while (ind < N) {
        printf("%f ", array[ind]);
        ind += blockDim.x * gridDim.x;
    }
}

__global__ void norm_gpu(int N)
{
    assert(blockDim.x * gridDim.x == N);
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

}

void preconditioner_gpu(float* b, float* x, int L, int d, float errtol)
{
    int N = pow(L,d);
    float* r = cuda_allocate_field_f(N);
    cudaMemcpy(r,b,N*sizeof(float),cudaMemcpyHostToDevice);
    float* tmp = cuda_allocate_field_f(N); //tmp field
    cudaMemcpy(tmp,r,N*sizeof(float),cudaMemcpyDeviceToDevice);
        
}




int main()
{
    run_tests_cpu();
    int L = 5;
    int d = 3;
    int N = (int)pow(L,d);
    float* b = cuda_allocate_field_f(N);
    float* x = cuda_allocate_field_f(N);

    preconditioner_gpu(b,x,L,d,1e-3);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
