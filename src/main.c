#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>


#define L 100
#define d 1
#define N pow(L,d)
// int L = 100;
// int d = 2;
// int N = pow(L,d);

float f2(float x, float y) {
    return (x * x)/2;
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
//     // double* (*laplace)[size] = malloc(sizeof(double[size][size]));
//     // double* laplace = malloc(size*size*sizeof(double));
//     double laplace[size][size];
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; i < size; j++) {
//             if (i == 0) {
//                 // todo
//             } else if (i == size - 1) {
//                 // todo
//             } else {
//                 laplace[i][j] = (f[i + 1][j] - 2 * f[i][j] + f[i - 1][j]) / (dx * dx) + (f[i][j + 1] - 2 * f[i][j] + f[i][j - 1]) / (dx * dx);
//             }
//         }
//     }
//     return laplace;
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
    int ind = 0;
    for (int i=0; i<d; i++) {
        // todo unedfined reference to pow
        ind += pow(L,i)*cords[i];
    }
    return ind;
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

double* laplace(double* u, double dx) {
    double* ddf = malloc(N*sizeof(double));
    for (int ny=0; ny < L; ny++)
    {
        for (int nx=0; nx < L; nx++)
        {
            int* cords = malloc(2*sizeof(int));
            cords[0] = nx;
            cords[1] = ny;
            // this index is sus
            int ind = get_index(cords);
            // todo
            // // this might segfault
            // // also it does not seem to calculate correctly
            // out[ind] = ((u[ind+1] - 2* u[ind] + u[ind+1]) + (u[ind+L] - 2* u[ind] + u[ind-L]))/(dx*dx);
            // // u[get_index((nx,ny))] = (u[get_index((nx+1,ny))] - 2* u[get_index((nx,ny))] + u[get_index((nx+1,ny))] - u[get_index((nx,ny))]) + (u[get_index((nx,ny+1))] - 2* u[get_index((nx,ny))] + u[get_index((nx,ny+1))] - u[get_index((nx,ny))]);
            // // u[get_index((nx+1,ny))] - 2* u[get_index((nx,ny))] + u[get_index((nx+1,ny))] - u[get_index((nx,ny))];
            if (ind == 0) {
            ddf[ind] = (u[ind + 2] - 2 * u[ind + 1] + u[ind]) / (dx * dx);
            } else if (ind == L - 1) {
                ddf[ind] = (u[ind] - 2 * u[ind - 1] + u[ind - 2]) / (dx * dx);
            } else {
                ddf[ind] = (u[ind + 1] - 2 * u[ind] + u[ind - 1]) / (dx * dx);
            } 
            printf("%d %d -> %d, %f\n", cords[0], cords[1], ind, ddf[ind]);
            free(cords);
        }
    }
    return ddf;
}


int main() {
    test_2nd_derivative();

    int i;
    double dx = 2.0 / (L - 1);
    double x, y;
    double* u = (double*) malloc(N*sizeof(float));

    for (i = 0; i < N; i++) {
        // x,y = index_to_cords(i);
        int nx = index_to_cords(i)[0];
        int ny = index_to_cords(i)[1];
        x = -1 + nx * dx;
        y = -1 + ny * dx;
        u[i] = f2(x, y);
    }
    laplace(u,dx);
    for (i = 0; i < N; i++) {
        printf("%f\n", u[i]);
    }
    return 0;
}
