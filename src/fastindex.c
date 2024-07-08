#include <stdio.h>

unsigned int generate_mask(int d)
{
    unsigned int m = 0;
    for (int i=0; i<64; i+=d)
    {
        m++;
        m << 1;
    }
    return m;
}

int main()
{
    printf("mask: %d", generate_mask(2));
}
