#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#define L 100  // Lattice size
#define d 10  // Dimension
#define N (int)pow(L,d) // Number of lattice points

float f2(float x, float y) {
    return (x * x + y*y)/2;
}

double f(double x)
{
    return x*x/2;
}

double* center_diff(double* f, int size, double dx) {
    double* df = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        if (i == 0) {
            df[i] = (f[i + 1] - f[i]) / dx;
        } else if (i == size - 1) {
            df[i] = (f[i] - f[i - 1]) / dx;
        } else {
            df[i] = (f[i + 1] - f[i - 1]) / (2 * dx);
        }
    }
    return df;
}

double* forward_diff(double* f, int size, double dx) {
    double* df = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        if (i == size - 1) {
            df[i] = (f[i] - f[i - 1]) / dx;
        } else {
            df[i] = (f[i + 1] - f[i]) / dx;
        }
    }
    return df;
}

double* second_derivative(double* f, int size, double dx) {
    double* ddf = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        if (i == 0) {
            ddf[i] = (f[i + 2] - 2 * f[i + 1] + f[i]) / (dx * dx);
        } else if (i == size - 1) {
            ddf[i] = (f[i] - 2 * f[i - 1] + f[i - 2]) / (dx * dx);
        } else {
            ddf[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / (dx * dx);
        }
    }
    return ddf;
}

// double** laplace_2d(int size, double dx, double f[size][size]) {
//     // double* (*minus_laplace)[size] = malloc(sizeof(double[size][size]));
//     // double* minus_laplace = malloc(size*size*sizeof(double));
//     double minus_laplace[size][size];
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; i < size; j++) {
//             if (i == 0) {
//                 // todo
//             } else if (i == size - 1) {
//                 // todo
//             } else {
//                 minus_laplace[i][j] = (f[i + 1][j] - 2 * f[i][j] + f[i - 1][j]) / (dx * dx) + (f[i][j + 1] - 2 * f[i][j] + f[i][j - 1]) / (dx * dx);
//             }
//         }
//     }
//     return minus_laplace;
// }

int test_2nd_derivative() {
    int i, j;
    int n = 1000;
    double dx = 2.0 / (n - 1);
    double x, y;
    double u[n];

    for (i = 0; i < n; i++) {
        x = -1 + i * dx;
        u[i] = f(x);
    }
    double* df = center_diff(u, n, dx);
    double* ddf = second_derivative(u, n, dx);
    for (i = 0; i < n; i++) {
        assert(fabs(ddf[i] - 1.0) < 1e-6);
    }
    return 0;
}

int get_index(int cords[d]) {
    for (int c = 0; c < d; c++)
    {
        int cord = cords[c];
        assert(cord < L+1 && cord > -2);
        if (cord==-1 || cord==L)
        {
            return N;
        }
    }

    int ind = 0;
    for (int i=0; i<d; i++) {
        ind += pow(L,i)*cords[i];
    }
    return ind;
}


int neighbour_index(int cords[d], int direction, int amount)
{
    int copy_cords[d];
    memcpy(copy_cords , cords, d*sizeof(int));
    copy_cords[direction] += amount;
    return get_index(copy_cords);
}

int* index_to_cords(int*cords, int index) {
    for ( int i=0; i<d; i++)
    {
        cords[i] = index % L;
        index /= L;
    }
    return cords;
}

double* minus_laplace(double* ddf, double* u, double dx) {
    for (int ind = 0; ind < N; ind++) {
            int cords[d];
            index_to_cords(cords,ind);
            float laplace_value = 0;
            for (int i=0; i<d; i++)
            {
                laplace_value += -u[neighbour_index(cords,i,1)] + 2* u[neighbour_index(cords, i, 0)] - u[neighbour_index(cords, i, -1)];

            } 
            ddf[ind] = laplace_value/pow(dx,d);
            
    }
    return ddf;
}

double norm(double* x,int n) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }
    return sqrt(norm);
}

double inner_product(double* x, double* y, int n) {
    double product = 0;
    for (int i = 0; i < n; i++) {
        product += x[i] * y[i];
    }
    return product;
}

void print_matrix(double* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

double* allocate_field()
{
    double* r = calloc(N+1,sizeof(double));
    if (r == NULL)
    {
        printf("Memory allocation failed");
        exit(1);
    }
    r[N] = 0;
    return r;
}

double* conjugate_gradient(double* b, double* x) {
    double* r = allocate_field();
    double* Ax = allocate_field();
    minus_laplace(Ax, x, 2.0 / (L - 1));
    printf("Ax:\n");
    print_matrix(Ax, L);
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - Ax[i];
    }
    double tol = 1e-3;
    // p = r;
    double* p = allocate_field();
    for (int i = 0; i < N; i++) {
        p[i] = r[i];
    }
    double* tmp = allocate_field();
    double* r_new = allocate_field();
    float dx = 2.0/(L-1);
    while (norm(r, N) > tol)
    {
        double* Ap =  minus_laplace(tmp,p, dx);
        double dx = 2.0 / (L - 1);
        double alpha = inner_product(r, r, N) / inner_product(p, Ap, N);
        for (int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
            r_new[i] = r[i] - alpha * Ap[i];
        }
        double beta = inner_product(r_new, r_new, N) / inner_product(r, r, N);
        for (int i = 0; i < N; i++) {
            p[i] = r_new[i] + beta * p[i];
            r[i] = r_new[i];
        }
    }
    free(tmp);
    free(r);
    free(r_new);
    free(Ax);
    free(p);
    return x;
}

bool every_f(double* x, double y, int n)
{
    double tol = 1e-3;
    for (int i = 0; i < n; i++)
    {
        if (x[i] - y > tol || x[i] - y < -tol) // floating point comparison
        {
            return false;
        }
    }
    return true;
}

bool every_a(double* x, double* y, int n)
{
    double tol = 1e-3;
    for (int i = 0; i < n; i++)
    {
        if (x[i] - y[i] > tol || x[i] - y[i] < -tol) // floating point comparison
        {
            return false;
        }
    }
    return true;
}

// TODO
int test_cg() {
    int i;
    double* x = allocate_field();
    double* b = allocate_field();
    for (i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = 1;
    }
    x = conjugate_gradient(b, x);
    print_matrix(x, L);
    double* Ax = allocate_field();
    minus_laplace(Ax, x, 2.0 / (L - 1));
    printf("--------------Test result: Ax should be 1 -----------------\n");
    print_matrix(Ax, L);
    assert(every_a(Ax, b, N));
    free(x);
    free(b);
    free(Ax);
    return 0;
}

int main() {
    test_2nd_derivative();

    int i;
    double dx = 2.0 / (L - 1);
    double x, y;
    double* u = allocate_field();
    u[N] = 0;
    for (i = 0; i < N; i++) {
        int cords[2];
        index_to_cords(cords,i);
        int nx = cords[0];
        int ny = cords[1];
        x = -1 + nx * dx;
        y = -1 + ny * dx;
        u[i] = f2(x, y);
    }
    double* ddf = allocate_field();
    ddf = minus_laplace(ddf, u,dx);
    free(u);
    free(ddf);

    test_cg();
    return 0;
}
