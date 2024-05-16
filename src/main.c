#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define L 10
#define d 2
#define N (int)pow(L,d)
// int L = 100;
// int d = 2;
// int N = pow(L,d);

float f2(float x, float y) {
    return (x * x + y*y)/2;
}

double f(double x) //why does this conflict with the other f function? They have different signatures?
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
    // double u[n][n];
    double u[n];

    for (i = 0; i < n; i++) {
        x = -1 + i * dx;
        u[i] = f(x);
    }
    double* df = center_diff(u, n, dx);
    double* ddf = second_derivative(u, n, dx);
    for (i = 0; i < n; i++) {
        // printf("%f\n", ddf[i]);
        // printf("%f\n", fabs(ddf[i] - 1.0));
        assert(fabs(ddf[i] - 1.0) < 1e-6);
    }
    return 0;
}

// get_index((1,0)) -> 20
// get_index((0,1)) -> 20
// this is not good
// todo
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

int* index_to_cords(int index) {
    int* cords = malloc(d*sizeof(int));
    for ( int i=0; i<d; i++)
    {
        cords[i] = index % L;
        index /= L;
    }
    return cords;
}

double* minus_laplace(double* u, double dx) {
    double* ddf = malloc(N*sizeof(double));
    for (int ny=0; ny < L; ny++)
    {
        for (int nx=0; nx < L; nx++)
        {
            int* cords = malloc(d*sizeof(int));
            cords[0] = nx;
            cords[1] = ny;
            int ind = get_index(cords);
            // TODO I am ignoring other boudnary conditions for now.
            // TODO Derivatives along the other directions do nto work yet.
            float laplace_value = 0;
            for (int i=0; i<d; i++)
            {
                laplace_value += -u[neighbour_index(cords,i,1)] + 2* u[neighbour_index(cords, i, 0)] - u[neighbour_index(cords, i, -1)];

            } 
            // printf("%d %d -> %d, %f\n", cords[0], cords[1], ind, ddf[ind]);
            // ddf[ind] = laplace_value/pow(dx,d);
            // printf("%f", laplace_value*1000);
            // printf("%f", laplace_value);
            // printf("%f", laplace_value/pow(dx,d));
            // printf("%i", ind);
            ddf[ind] = laplace_value/pow(dx,d);
            free(cords);
        }
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


double* conjugate_gradient(double* b, double* x) {
    // double* r = b - A*x;
    double* r = malloc(N*sizeof(double));
    double* Ax = minus_laplace(x, 2.0 / (L - 1));
    printf("Ax:\n");
    print_matrix(Ax, L);
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - Ax[i];
    }
    double tol = 1e-6;
    // double* p = r;
    double* p = malloc(N*sizeof(double));
    for (int i = 0; i < N; i++) {
        p[i] = r[i];
    }
    while (norm(r, N) > tol)
    {

        double dx = 2.0 / (L - 1);
        double alpha = inner_product(r, r, N) / inner_product(p, minus_laplace(p, dx), N);
        double* r_new = malloc(N*sizeof(double));
        for (int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
            r_new[i] = r[i] - alpha * minus_laplace(p, dx)[i];
        }
        double beta = inner_product(r_new, r_new, N) / inner_product(r, r, N);
        for (int i = 0; i < N; i++) {
            p[i] = r_new[i] + beta * p[i];
            r[i] = r_new[i];
        }
    }

    return x;
}

int test_cg() {
    int i;
    double* x = (double*) malloc(N*sizeof(double));
    double* b = (double*) malloc(N*sizeof(double));
    for (i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = 1;
    }
    x = conjugate_gradient(b, x);
    print_matrix(x, L); // upper boundary is -inf
    double* Ax = minus_laplace(x, 2.0 / (L - 1));
    print_matrix(Ax, L);
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
    double* u = (double*) malloc((N+1)*sizeof(float));
    u[N] = 0;
    for (i = 0; i < N; i++) {
        // x,y = index_to_cords(i);
        int nx = index_to_cords(i)[0];
        int ny = index_to_cords(i)[1];
        x = -1 + nx * dx;
        y = -1 + ny * dx;
        u[i] = f2(x, y);
    }
    double* ddf = minus_laplace(u,dx);
    // for (i = 0; i < N; i++) {
    //     printf("%f\n", ddf[i]);
    // }
    for (i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            printf("%f ", ddf[get_index((int[]){i,j})]);
        }
        printf("\n");
    }
    free(u);
    free(ddf);

    // test_cg();
    return 0;
}
