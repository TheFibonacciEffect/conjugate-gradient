#include <stdio.h>

__global__ void my_kernel(unsigned arr1_sz, unsigned arr2_sz){

  extern __shared__ char array[];

  double *my_ddata = (double *)array;
  char *my_cdata = arr1_sz*sizeof(double) + array;

  for (int i = 0; i < arr1_sz; i++) my_ddata[i] = (double) i*1.1f;
  for (int i = 0; i < arr2_sz; i++) my_cdata[i] = (char) i;

  printf("at offset %d, arr1: %lf, arr2: %d\n", 10, my_ddata[10], (int)my_cdata[10]);
}

int main(){
  unsigned double_array_size = 256;
  unsigned char_array_size = 128;
  unsigned shared_mem_size = (double_array_size*sizeof(double)) + (char_array_size*sizeof(char));
  my_kernel<<<1,1, shared_mem_size>>>(256, 128);
  cudaDeviceSynchronize();
  return 0;
}