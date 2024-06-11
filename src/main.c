#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

// #define L 5  // Lattice size
// #define d 3  // Dimension
// #define N (int)pow(L,d) // Number of lattice points

#include "conjugate-gradient_cpu.h"

#define TYPE double
#include "laplace-x.h"
#define TYPE float
#include "laplace-x.h"

int main()
{
    int L = 5;
    int d = 3;
    int N = (int)pow(L,d);
    double* b = cuda_allocate_field(N);
}
