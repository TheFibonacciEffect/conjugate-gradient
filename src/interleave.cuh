#pragma once
#include <array>
#include <assert.h>
#include <stdio.h>

unsigned int pdep_custom(unsigned int, unsigned int);
unsigned int pext_custom(unsigned int, unsigned int);

namespace Detail
{

  using index_t = unsigned int;

  constexpr unsigned int generate_mask(int d)
  {
    unsigned int m = 0;
    for (int i = 0; i < 32; i += d)
    {
      m = m << d;
      m++;
    }
    return m;
  }

  template <int Dim>
  struct mask_gen
  {
    static constexpr index_t value = generate_mask(Dim);
  };

  template <int N>
  __host__ __device__ static inline constexpr index_t generate_mask(int dim)
  {
    index_t i = mask_gen<N>::value;
    return i << dim;
  }

  template <int v, int N>
  struct NH
  {
  };

  template <int N>
  struct NH<-1, N>
  {
    __host__ __device__ static index_t calc(index_t arg, int dim)
    {
      index_t mask = generate_mask<N>(dim);
      index_t iy = arg & ~mask;
      index_t ix = arg & mask;
      index_t adj = ix - 1;
      return iy | (adj & mask);
    }
  };

  template <int N>
  struct NH<1, N>
  {
    __host__ __device__ static index_t calc(index_t arg, int dim)
    {
      index_t mask = generate_mask<N>(dim);
      index_t iy = arg & ~mask;
      index_t ix = (arg & mask) | ~mask;
      index_t adj = ix + 1;
      return iy | (adj & mask);
    }
  };
} // namespace Detail

template <int N>
struct Coords
{
  std::array<int, N> coords;

  int &operator[](int i) { return coords[i]; }
};

template <int N>
struct Index
{
  Detail::index_t i;

  __host__ Coords<N> to_coords()
  {
    Coords<N> res;
    for (int j = 0; j < N; j++)
    {
      res.coords[j] = pext_custom(i, Detail::generate_mask<N>(j));
    }
    return res;
  }

  template <int Dir>
  __host__ __device__ Index<N> neighbour(int dim)
  {
    return {Detail::NH<Dir, N>::calc(i, dim)};
  }

  __host__ bool is_outside(unsigned int size) { return i >= size; }
};

template <typename T, int N>
struct PtrND
{
  T *t;

  __host__ __device__ T &operator[](Index<N> i) { return t[i.i]; }
};

template <int N>
__host__ static inline Index<N>
from_coords(Coords<N> c)
{
  Detail::index_t res = 0;
  for (int i = 0; i < N; i++)
    res |= pdep_custom(c.coords[i], Detail::generate_mask<N>(i));
  return {res};
}

int tests_interleaved_index()
{
  Coords<2> c({2, 3});
  Index<2> i = from_coords(c);
  auto i2 = i.neighbour<1>(1);
  auto i3 = i.neighbour<1>(0);
  auto i4 = i.neighbour<-1>(1);
  auto i5 = i.neighbour<-1>(0);
  auto c_back = i.to_coords();
  auto c2 = i2.to_coords();
  auto c3 = i3.to_coords();
  auto c4 = i4.to_coords();
  auto c5 = i5.to_coords();
  int *p = (int *)malloc(sizeof(int) * 100);
  PtrND<int, 2> ptr{p};
  ptr[i] = 10;
  printf("%#x = %d, %d\n", i.i, c_back[0], c_back[1]);
  printf("%#x = %d, %d\n", i2.i, c2[0], c2[1]);
  printf("%#x = %d, %d\n", i3.i, c3[0], c3[1]);
  printf("%#x = %d, %d\n", i4.i, c4[0], c4[1]);
  printf("%#x = %d, %d\n", i5.i, c5[0], c5[1]);
  printf("ptr[i] = %d\n", ptr[i]);
  assert(c_back[0] == 2);
  assert(c_back[1] == 3);

  Coords<2> cover({0, 0});
  auto iover = from_coords(cover);
  auto ioverleft = iover.neighbour<-1>(0);
  auto coverleft = ioverleft.to_coords();
  printf("%#x = %d %d (outside? %d)\n", ioverleft.i, coverleft[0], coverleft[1],
         ioverleft.is_outside(16 * 16));

  return 0;
}