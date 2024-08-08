#pragma once // helps with doubble includes
#include <stdio.h>
#include "common.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>

__host__ float *cuda_allocate_field(int N);

__device__ int *index_to_cords_cu(int *cords, int index, int L, int d);

__device__ int get_index_gpu(int *cords, int L, int d, int N);

extern "C" __host__ __device__ int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
                    int N, int index_mode);

/* index_mode:
0 = naiive
1 = bit mixing
2 = lookup table */
__global__ void laplace_gpu(float *ddf, float *u, int d,
                                   int L, int N, unsigned int index_mode);

__global__ void reduceMulAddComplete(float *v, float *w, float *g_odata,
                                            unsigned int n,const unsigned int nthreads);

extern "C" float inner_product_gpu(float *v, float *w, unsigned int N);

__host__ float norm(float *v, int N);

// TODO still work in progress
// initial guess is 0
extern "C" float conjugate_gradient_gpu(float * b, float * x , int L, int d);