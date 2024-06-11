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


