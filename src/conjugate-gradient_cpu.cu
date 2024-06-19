#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>


float square(float x, float y) {
    return (x * x + y*y)/4;
}


int get_index(int* cords, int L, int d, int N) {
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


int neighbour_index(int* cords, int direction, int amount, int L, int d, int N)
{
    int copy_cords[d];
    memcpy(copy_cords , cords, d*sizeof(int));
    copy_cords[direction] += amount;
    return get_index(copy_cords, L,d,N);
}

int* index_to_cords(int*cords, int index, int L, int d) {
    assert(index < pow(L,d) && index >= 0);
    for ( int i=0; i<d; i++)
    {
        cords[i] = index % L;
        index /= L;
    }
    return cords;
}

double* minus_laplace(double* ddf, double* u, double dx, int d, int L, int N) {
    for (int ind = 0; ind < N; ind++) {
            int cords[d];
            index_to_cords(cords,ind, L, d);
            double laplace_value = 0;
            for (int i=0; i<d; i++)
            {
                laplace_value += -u[neighbour_index(cords,i,1, L,d,N)] + 2* u[neighbour_index(cords, i, 0, L,d,N)] - u[neighbour_index(cords, i, -1, L,d,N)];

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

double* allocate_field(int N)
{
    double* r = (double *) calloc(N+1,sizeof(double));
    if (r == NULL)
    {
        printf("Memory allocation failed");
        exit(1);
    }
    r[N] = 0;
    return r;
}

double* conjugate_gradient(double* b, double* x, int L, int d) {
    int N = pow(L,d);
    double* r = allocate_field(N);
    double* Ax = allocate_field(N);
    minus_laplace(Ax, x, 2.0 / (L - 1), d, L, N);
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - Ax[i];
    }
    double tol = 1e-3;
    // p = r;
    double* p = allocate_field(N);
    for (int i = 0; i < N; i++) {
        p[i] = r[i];
    }
    double* tmp = allocate_field(N);
    double* r_new = allocate_field(N);
    float dx = 2.0/(L-1);
    while (norm(r, N) > tol)
    {
        double* Ap =  minus_laplace(tmp,p, dx, d, L, N);
        // double dx = 2.0 / (L - 1);
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

double* random_array(double* r, int L, int d, int N)
{
    // double* r = allocate_field(N);
    for (int i = 0; i < N; i++)
    {
        r[i] = (double)rand() / (double)RAND_MAX;
    }
    return r;
}

bool test_cg(int L, int d, int N) {
    // allocate an array
    double dx = 2.0 / (L - 1);
    double* x = allocate_field(N);
    double* b = allocate_field(N);
    // fill it with random data
    random_array(x, L, d, N);
    // calculate b
    b = minus_laplace(b, x, dx, d, L, N);
    // initial guess
    double* x0 = allocate_field(N);
    for (int i = 0; i < N; i++) {
        x0[i] = 0;
    }
    // apply conjugate gradient to calculate x
    x0 = conjugate_gradient(b, x0, L, d);
    // compare with x
    bool passed = false;
    if (every_a(x, x0, N))
    {
        passed = true;
    }
    else
    {
        passed = false;
    }
    free(x);
    free(b);
    free(x0);
    return passed;
}

bool is_boundary(int* cords, int L, int d) {
    for (int i = 0; i < d; i++) {
        if (cords[i] == 0 || cords[i] == L-1) {
            return true;
        }
    }
    return false;
}

bool test_laplace() {
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int i;
    double dx = 2.0 / (L - 1);
    double x, y;
    double* u = allocate_field(N);
    u[N] = 0;
    for (i = 0; i < N; i++) {
        int cords[2];
        index_to_cords(cords,i, L, d);
        int nx = cords[0];
        int ny = cords[1];
        x = -1 + nx * dx;
        y = -1 + ny * dx;
        u[i] = square(x, y);
    }
    double* ddf = allocate_field(N);
    ddf = minus_laplace(ddf, u,dx, d, L, N);

    for (int i = 0; i < N; i++) {
        int cords[2];
        index_to_cords(cords,i, L, d);
        int nx = cords[0];
        int ny = cords[1];
        x = -1 + nx * dx;
        y = -1 + ny * dx;
        if (!is_boundary(cords,L,d) && !(ddf[i] - 1 > 1e-3 || ddf[i] - 1 < -1e-3)) {
            printf("Test failed\n");
            printf("x: %f, y: %f, ddf: %f\n", x, y, ddf[i]);
            return false;
        }
    }
    free(u);
    free(ddf);
    return true;

}

bool run_test_gc_cpu() {
    int L = 5;
    int d = 3;
    int N = pow(L,d);
    test_cg( L, d, N);
    return true;
}

bool test_inner_product() {
    double x[3] = {1, 2, 3};
    double y[3] = {4, 5, 6};
    double result = inner_product(x, y, 3);
    if (result - 32 > 1e-3 || result - 32 < -1e-3) {
        return false;
    }
    return true;
}

bool test_norm() {
    double x[3] = {1, 2, 3};
    double result = norm(x, 3);
    if (result - sqrt(14) > 1e-3 || result - sqrt(14) < -1e-3) {
        return false;
    }
    return true;
}

bool test_getindex() {
    int cords[2] = {1, 2};
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int result = get_index(cords, L, d, N);
    if (result != 11) {
        return false;
    }
    return true;
}

bool test_getindex_edge() {
    int cords[2] = {5, 5}; // outside the array
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int result = get_index(cords, L, d, N);
    if (result != N) {
        return false;
    }
    return true;
}

bool test_getindex_edge2() {
    int cords[2] = {-1, 0}; // outside the array
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int result = get_index(cords, L, d, N);
    if (result != N) {
        return false;
    }
    return true;
}

bool test_neighbour_index() {
    int cords[2] = {0, 2};
    int direction = 1;
    int amount = -1;
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int result = neighbour_index(cords, direction, amount, L, d, N);
    if (result != 5) {
        return false;
    }
    return true;
}

bool test_neighbour_index2() {
    int cords[2] = {1, 2};
    int direction = 1;
    int amount = -1;
    int L = 5;
    int d = 2;
    int N = pow(L,d);
    int result = neighbour_index(cords, direction, amount, L, d, N);
    if (result != 6) {
        return false;
    }
    return true;
}

extern int run_tests_cpu() {
    assert(test_laplace());
    assert(run_test_gc_cpu());
    assert(test_inner_product());
    assert(test_norm());
    assert(test_getindex());
    assert(test_getindex_edge2());
    assert(test_getindex_edge());
    assert(test_neighbour_index());
    printf("Tests Passed!\n");
    return 0;
}