#include <stdio.h>
#include <stdlib.h>
int * alloc(int n)
{
    int* A = calloc(n, sizeof(int));
    return A;
}


void free_mem(int *A)
{
    free(A);
}

void print_array(int *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");
}


