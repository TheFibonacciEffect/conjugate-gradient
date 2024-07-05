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

#include "laplace-d.h"
#include "preconditioner_gpu.h"

int main()
{
    run_tests_cpu();
    int L = 5;
    int d = 3;
    int N = (int)pow(L,d);
    double* b = cuda_allocate_field_d(N);
    double* x = cuda_allocate_field_d(N);

    // TODO Undefined reference
    preconditioner_gpu(b,x,L,d,1e-3);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
