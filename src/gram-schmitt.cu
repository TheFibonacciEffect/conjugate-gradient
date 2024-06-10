#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cassert>
#include "common.h"
#define NTHREADS 32
// #include <stdlib.h>
// #include <float.h>
__global__ void sum(int* g_idata, int* g_odata)
{
   __shared__ int sdata[NTHREADS]; //why do I not need to reserve this?

   // each itteration loads one element from global to shared memory
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   sdata[tid] = g_idata[i];
   __syncthreads();

   // do reduction in shared memory
   // why cant I do it in the normal memory (whatever that is)?
   for (unsigned int s = 1; s < blockDim.x; s*=2)
   {
      if (tid % (2*s) == 0)
      {
         sdata[tid] += sdata[tid +s];
      }
      __syncthreads();
   }
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



__global__ void fill(int* data, int value, const int N)
{
   int idx;
   idx=blockIdx.x*blockDim.x + threadIdx.x;
   for (; (idx<N); idx+=blockDim.x*gridDim.x)
      data[idx] = value;
}

int main(){
   int N = 10000;
   int nblocks = 32;
   int *A, *zwischenergebnisse, *ergebnis, *result_cpu;
   CHECK(cudaMalloc(&A,                   N*sizeof(int)));
   CHECK(cudaMalloc(&zwischenergebnisse,  nblocks*sizeof(int)));
   CHECK(cudaMalloc(&ergebnis,            sizeof(int)));
   fill<<<nblocks,NTHREADS>>>(A,1,N);
   cudaDeviceSynchronize();
   sum<<<nblocks,NTHREADS>>>(A,zwischenergebnisse);
   cudaDeviceSynchronize();
   assert(nblocks <= NTHREADS);
   sum<<<1,nblocks>>>(zwischenergebnisse, ergebnis);
   cudaDeviceSynchronize();
   result_cpu = (int*) malloc(sizeof(int));
   if (result_cpu == NULL) {
      printf("allocation failed"); 
      exit(1);
   }

   CHECK(cudaMemcpy(result_cpu, ergebnis, sizeof(int), cudaMemcpyDeviceToHost));
   printf("%d\n", *result_cpu);
}
