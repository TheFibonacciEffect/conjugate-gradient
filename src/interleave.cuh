#pragma once
#include <array>
#include <assert.h>
#include <stdio.h>

unsigned int pdep_custom(unsigned int, unsigned int);
unsigned int pext_custom(unsigned int, unsigned int);

namespace Detail {

  using index_t = unsigned int;

  template<int Dim>
  struct mask_gen {};

  template<>
  struct mask_gen<2> {
	static constexpr unsigned int value = 0x55555555;
  };

  template<>
  struct mask_gen<3> {
	static constexpr unsigned int value = 0x24924924;
  };

  template<>
  struct mask_gen<4> {
	static constexpr unsigned int value = 0x44444444;
  };

  template<>
  struct mask_gen<5> {
	static constexpr unsigned int value = 0x21084210;
  };

  template<>
  struct mask_gen<6> {
	static constexpr unsigned int value = 0x20820820;
  };

  template<int N>
  __host__ __device__ static inline constexpr index_t generate_mask(int dim) {
	index_t i = mask_gen<N>::value;
	return i << dim;
  }

  template<int v, int N>
  struct NH {};

  template<int N>
  struct NH<-1, N> {
	__host__ __device__ static index_t calc(index_t arg, int dim) {
	  index_t mask = generate_mask<N>(dim);
	  index_t iy = arg & ~mask;
	  index_t ix = arg & mask;
	  index_t adj = ix - 1;
	  return iy | (adj & mask);
	}
  };

  template<int N>
  struct NH<1, N> {
	__host__ __device__ static index_t calc(index_t arg, int dim) {
	  index_t mask = generate_mask<N>(dim);
	  index_t iy = arg & ~mask;
	  index_t ix = (arg & mask) | ~mask;
	  index_t adj = ix + 1;
	  return iy | (adj & mask);
	}
  };
}// namespace Detail

template<int N>
struct Coords {
  std::array<int, N> coords;

  int &operator[](int i) { return coords[i]; }
};

template<int N>
struct Index {
  Detail::index_t i;

  __host__ Coords<N> to_coords() {
	Coords<N> res;
	for (int j = 0; j < N; j++) {
	  res.coords[j] = pext_custom(i, Detail::generate_mask<N>(j));
	}
	return res;
  }

  template<int Dir>
  __host__ __device__ Index<N> neighbour(int dim) {
	return {Detail::NH<Dir, N>::calc(i, dim)};
  }

  __host__ bool is_outside(unsigned int size) { return i >= size; }
};

template<typename T, int N>
struct PtrND {
  T *t;

  __host__ __device__ T &operator[](Index<N> i) { return t[i.i]; }
};

template<int N>
__host__ static inline Index<N>
from_coords(Coords<N> c) {
  Detail::index_t res = 0;
  for (int i = 0; i < N; i++)
	res |= pdep_custom(c.coords[i], Detail::generate_mask<N>(i));
  return {res};
}

