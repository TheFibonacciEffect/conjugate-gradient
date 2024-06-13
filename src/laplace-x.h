#ifndef UTILS_GPU_X_H
#define UTILS_GPU_X_H

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


__device__ int* index_to_cords(int*cords, int index, int L, int d) {
    assert(index < pow(L,d) && index >= 0);
    for ( int i=0; i<d; i++)
    {
        cords[i] = index % L;
        index /= L;
    }
    return cords;
}

# endif

__global__ void FUNCTION(apply_function_gpu)(TYPE * result, TYPE (*f)(int), int N, int L, int d) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int cords[dmax];
        index_to_cords(cords, ind,  L, d);
        result[ind] = f(cords[0]);
    }
}

__host__ TYPE* FUNCTION(cuda_allocate_field)(int N) {
    TYPE * field;
    CHECK(cudaMallocManaged(&field, (N+1)*sizeof(TYPE)));
    field[N] = 0;
    return field;
}


// this is a type agnostic version of the laplace operator and associated functions on the gpu
__global__ void FUNCTION(minus_laplace_gpu_)(TYPE * ddf, TYPE * u, TYPE dx, int d, int L, int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int cords[dmax];
        index_to_cords(cords,ind, L, d);
        TYPE laplace_value = 0;
        for (int i = 0; i < d; i++)
        {
            laplace_value += -u[neighbour_index_gpu(cords,i,1, L,d,N)] + 2* u[neighbour_index_gpu(cords, i, 0, L,d,N)] - u[neighbour_index_gpu(cords, i, -1, L,d,N)];
        }
        ddf[ind] = laplace_value/pow(dx,d);
    }
}
