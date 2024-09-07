#include <immintrin.h>
#include <x86intrin.h>

unsigned int
pdep_custom(unsigned int x, unsigned int y)
{
  return _pdep_u32(x, y);
}
unsigned int
pext_custom(unsigned int x, unsigned int y)
{
  return _pext_u32(x, y);
}
