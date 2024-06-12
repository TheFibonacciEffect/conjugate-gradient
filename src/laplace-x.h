# ifndef UTILS_GPU_X_H
# define UTILS_GPU_X_H

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
    return ind
}

# endif


__global__ void apply_function_gpu_##TYPE(TYPE * field, TYPE * result, TYPE (*f)(TYPE), int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        result[ind] = f(field[ind]);
    }
}

__host__ void cuda_allocate_field##TYPE(TYPE* field, int N) {
    TYPE * field;
    CHECK(cudaMallocManaged(&field, (N+1)*sizeof(TYPE)));
    field[N] = 0;
    return field;
}


// this is a type agnostic version of the laplace operator and associated functions on the gpu
__global__ void minus_laplace_gpu_##TYPE(TYPE * ddf, TYPE * u, TYPE dx, int d, int L, int N) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < N) {
        int cords[d];
        index_to_cords(cords,ind, L, d);
        TYPE laplace_value = 0;
        for (int i = 0; i < d; i++)
        {
            laplace_value += -u[neighbour_index_gpu(cords,i,1, L,d,N)] + 2* u[neighbour_index_gpu(cords, i, 0, L,d,N)] - u[neighbour_index_gpu(cords, i, -1, L,d,N)];
        }
        ddf[ind] = laplace_value/pow(dx,d);
    }
}
