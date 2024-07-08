#include <stdio.h>
#include <x86intrin.h>

typedef unsigned index_t;

const unsigned mask = 0x55555555;

index_t
enc_index(unsigned x, unsigned y) {
  unsigned ix = _pdep_u32(x, mask);
  unsigned iy = _pdep_u32(y, ~mask);
  return ix | iy;
}

void
dec_index(index_t index, unsigned *x, unsigned *y) {
  *x = _pext_u32(index, mask); 
  *y = _pext_u32(index, ~mask);
}

index_t
incx(index_t i) {
  unsigned iy = i & ~mask;
  unsigned ix = (i & mask) | ~mask;
  unsigned incremented = ix + 1;
  return iy | (incremented & mask);
}

index_t
addx(index_t i, unsigned delta) {
  unsigned deltap = _pdep_u32(delta, mask);
  unsigned iy = i & ~mask;
  unsigned ix = (i & mask) | ~mask;
  unsigned incremented = ix + deltap;
  return iy | (incremented & mask);
}

int
main(int argc, char **argv) {
  unsigned x, y;

  index_t i = enc_index(255, 255);  
  // index_t ip = incx(i);
  index_t ip = addx(i, 3);


  dec_index(ip, &x, &y);
  printf("%#x = %#x + 3e_x, x = %d, y = %d\n", ip, i, x, y);
  
  return 0;
}
