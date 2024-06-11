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
