// here are slightly altered functions for compiling with gcc to obtain their assembly code.
int index_to_cords(int index, int L, int d)
{
  for (int i = 0; i < d; i++)
  {
    index /= L;
  }
  return index % L;
}

int neighbour_index_gpu(int ind, int direction,
                                                       int amount, int L, int d,
                                                       int N, int index_mode)
{
  // should be consistant with cpu code
  int cord = index_to_cords(ind, L, direction);
  cord += amount;
  int n = 1;
  for (int i = 0; i < direction; i++)
  {
    n *= L;
  }
  ind += amount * n;
  return ind;
}

void laplace_gpu(float *ddf, float *u, int d, int L, int N,
                            unsigned int index_mode, int ind)
{
//   int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < N)
  {
    float laplace_value = 0;
    for (int i = 0; i < d; i++)
    {
      laplace_value += -u[neighbour_index_gpu(ind, i, 1, L, d, N, index_mode)] + 2 * u[neighbour_index_gpu(ind, i, 0, L, d, N, index_mode)] - u[neighbour_index_gpu(ind, i, -1, L, d, N, index_mode)];
    }
    // the discrete version is defined without dx
    ddf[ind] = laplace_value;
  }
}