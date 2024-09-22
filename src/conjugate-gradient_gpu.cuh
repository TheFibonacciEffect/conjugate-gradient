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

extern "C" __host__ __device__ unsigned int neighbour_index_gpu(unsigned int ind, int direction,
                                                       int amount, unsigned int L, unsigned int d,
                                                       unsigned int N, int index_mode);

__global__ void laplace_gpu(float *ddf, float *u, unsigned int d,
                            unsigned int L, unsigned int N, unsigned int index_mode);

__global__ void reduceMulAddComplete(float *v, float *w, float *g_odata,
                                     unsigned int n, const unsigned int nthreads);

extern "C" float inner_product_gpu(float *v, float *w, unsigned int N);

__host__ float norm(float *v, int N);

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
extern "C" float conjugate_gradient_gpu(float *b, float *x, int L, int d);

__global__ void fillArray(float *arr, float value, int size);
float *random_array(float *r, int L, int d, int N);