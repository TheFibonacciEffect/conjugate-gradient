#include <stdio.h>

constexpr unsigned int generate_mask(int d)
{
    unsigned int m = 0;
    for (int i=0; i<32; i+=d)
    {
        m = m << d;
        m++;
    }
    return m;
}

int main()
{
    printf("mask: %d\n", generate_mask(2)); // 0101 0101 0101 0101 0101 0101 0101 0101
    printf("mask: %d\n", generate_mask(3)); // 0100 1001 0010 0100 1001 0010 0100 1001
}
