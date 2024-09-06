#include <cassert>
extern "C" int neighbour_index_gpu(int ind, int direction, int amount, int L, int d,
                    int N, int index_mode)
{
  assert(amount == 1 || amount == -1 || amount == 0);
  int n=1;
  for (int i=0; i<direction; i++)
  {
      n *= L;
  }
  ind += amount*n;
  // TODO Is there hardware side bounds checking? (See intel MPX)
  if (ind % L == 0 || ind +1 % L == 0 || ind < 0 || ind > N){
    return N;
  }
  return ind;
}

int main()
{
    neighbour_index_gpu(2,0,1,3,2,9,0);
}