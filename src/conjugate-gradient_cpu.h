// extern "C" {
//     int run_test_gc_cpu();
// }

double *allocate_field(int N);
double *minus_laplace(double *ddf, double *u, int d, int L, int N);
double conjugate_gradient(const double *b, double *x, int L, int d);
double preconditioner(double *b, double *x, int L, int d, double errtol);
static inline __device__ __host__ int index_to_cords(int index, int L, int d);
double inner_product(const double *x, const double *y, int n);
double *preconditioned_cg(double *b, double *x, int L, int d);
int *index_to_cords(int *cords, int index, int L, int d);
double norm(double *x, int n);
int get_index(int *cords, int L, int d, int N);
int neighbour_index(int *cords, int direction, int amount, int L, int d, int N);
double *random_array(double *r, int L, int d, int N);