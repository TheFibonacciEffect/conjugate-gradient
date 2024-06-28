#include <stdio.h>
// #include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "common.h"

// #define L 5  // Lattice size
// #define N (int)pow(L,d) // Number of lattice points

#define dmax 5  // Dimension

#define TYPE double
#define FUNCTION(NAME) NAME ## _d
#include "laplace-x.h"
#undef TYPE
// #define TYPE float
// #include "laplace-x.h"

#include "conjugate-gradient_cpu.h"

int main()
{
    run_tests_cpu();
    int L = 5;
    int d = 3;
    int N = (int)pow(L,d);
    double* b = cuda_allocate_field_d(N);
    double* x = cuda_allocate_field_d(N);
    double* x_cpu = (double*)malloc(N*sizeof(double));
    apply_function_gpu_d<<<1000,128>>>(x,N,L,d);    
    cudaDeviceSynchronize();
    cudaMemcpy(x_cpu,x,N*sizeof(double),cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f\n",x_cpu[i]);
    // }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
