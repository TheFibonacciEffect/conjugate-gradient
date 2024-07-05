#include <stdio.h>
// #include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "common.h"
#include "laplace-d.h"



__global__ void norm_gpu(int N)
{
    assert(blockDim.x * gridDim.x == N);
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

}

void preconditioner_gpu(double* b, double* x, int L, int d, double errtol)
{
    int N = pow(L,d);
    double* r = cuda_allocate_field_d(N);
    cudaMemcpy(r,b,N*sizeof(double),cudaMemcpyHostToDevice);
    double* tmp = cuda_allocate_field_d(N); //tmp field
    cudaMemcpy(tmp,r,N*sizeof(double),cudaMemcpyDeviceToDevice);
        
}

