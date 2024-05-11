// https://www.gnu.org/software/gsl/doc/html/linalg.html#examples
#include <stdio.h>
#include <gsl/gsl_linalg.h>

int main(void)
{
    double a_data[] = {0.18, 0.60, 0.57, 0.96,
                       0.41, 0.24, 0.99, 0.58,
                       0.14, 0.30, 0.97, 0.66,
                       0.51, 0.13, 0.19, 0.85};

    double b_data[] = {1.0, 2.0, 3.0, 4.0};

    gsl_matrix_view m = gsl_matrix_view_array(a_data, 4, 4);

    gsl_vector_view b = gsl_vector_view_array(b_data, 4);

    gsl_vector *x = gsl_vector_alloc(4);

    int s;

    gsl_permutation *p = gsl_permutation_alloc(4);

    gsl_linalg_LU_decomp(&m.matrix, p, &s);

    gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);

    printf("x = \n");
    gsl_vector_fprintf(stdout, x, "%g");
    // int gsl_blas_dgemv(CBLAS_TRANSPOSE_t TransA, double alpha, const gsl_matrix *A, const gsl_vector *x, double beta, gsl_vector *y)
    gsl_blas_dgemv(CblasNoTrans, 1.0, &m.matrix, x, 0.0, x);
    gsl_vector_fprintf(stdout, x, "%g"); //this is not as expected, it just prints out 0.0 0.0 0.0 0.0
    gsl_permutation_free(p);
    gsl_vector_free(x);
    return 0;
}