// newton_schulz_pthreads.c
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

typedef double real;

typedef struct {
    int tid;
    int n;
    int row_start;
    int row_end;
    real *A;
    real *B;
    real *C;
} MatMulArgs;

static inline real idx(real *M, int n, int i, int j) { return M[i*n + j]; }
static inline real *ptr(real *M, int n, int i) { return &M[i*n]; }

void *matmul_worker(void *argp) {
    MatMulArgs *args = (MatMulArgs*)argp;
    int n = args->n;
    int rs = args->row_start;
    int re = args->row_end;
    real *A = args->A;
    real *B = args->B;
    real *C = args->C;

    for (int i = rs; i < re; ++i) {
        real *Ai = &A[i*n];
        real *Ci = &C[i*n];
        for (int j = 0; j < n; ++j) Ci[j] = 0.0;
        for (int k = 0; k < n; ++k) {
            real a = Ai[k];
            real *Bk = &B[k*n];
            for (int j = 0; j < n; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }
    return NULL;
}

void matmul_parallel(real *A, real *B, real *C, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    MatMulArgs *args = malloc(nthreads * sizeof(MatMulArgs));
    int base = n / nthreads;
    int rem = n % nthreads;
    int cur = 0;
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].tid = t;
        args[t].n = n;
        args[t].row_start = cur;
        args[t].row_end = cur + rows;
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
        cur += rows;
    }
    for (int t = 0; t < nthreads; ++t) pthread_join(threads[t], NULL);
    free(threads);
    free(args);
}

// parallel elementwise: C = alpha*X + beta*Y  (if Y==NULL then beta=0)
typedef struct {
    int row_start, row_end, n;
    real *X, *Y, *C;
    real alpha, beta;
    int op; // 0: C = alpha*X + beta*Y, 1: C = X - Y, 2: C = X + Y, 3: scale X->C (alpha*X)
} ElemArgs;

void *elem_worker(void *ap) {
    ElemArgs *a = (ElemArgs*)ap;
    int n = a->n;
    for (int i = a->row_start; i < a->row_end; ++i) {
        real *Xi = &a->X[i*n];
        real *Ci = &a->C[i*n];
        if (a->op == 0) {
            real *Yi = a->Y ? &a->Y[i*n] : NULL;
            for (int j = 0; j < n; ++j) {
                Ci[j] = a->alpha * Xi[j] + (Yi ? a->beta * Yi[j] : 0.0);
            }
        } else if (a->op == 1) {
            real *Yi = &a->Y[i*n];
            for (int j = 0; j < n; ++j) Ci[j] = Xi[j] - Yi[j];
        } else if (a->op == 2) {
            real *Yi = &a->Y[i*n];
            for (int j = 0; j < n; ++j) Ci[j] = Xi[j] + Yi[j];
        } else if (a->op == 3) {
            for (int j = 0; j < n; ++j) Ci[j] = a->alpha * Xi[j];
        }
    }
    return NULL;
}

void elemwise_lincomb(real *X, real *Y, real *C, real alpha, real beta, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    ElemArgs *args = malloc(nthreads * sizeof(ElemArgs));
    int base = n / nthreads, rem = n % nthreads, cur = 0;
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].row_start = cur;
        args[t].row_end = cur + rows;
        args[t].n = n;
        args[t].X = X; args[t].Y = Y; args[t].C = C;
        args[t].alpha = alpha; args[t].beta = beta;
        args[t].op = 0;
        pthread_create(&threads[t], NULL, elem_worker, &args[t]);
        cur += rows;
    }
    for (int t=0;t<nthreads;++t) pthread_join(threads[t], NULL);
    free(threads); free(args);
}

void elemwise_sub(real *X, real *Y, real *C, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    ElemArgs *args = malloc(nthreads * sizeof(ElemArgs));
    int base = n / nthreads, rem = n % nthreads, cur = 0;
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].row_start = cur;
        args[t].row_end = cur + rows;
        args[t].n = n;
        args[t].X = X; args[t].Y = Y; args[t].C = C;
        args[t].op = 1;
        pthread_create(&threads[t], NULL, elem_worker, &args[t]);
        cur += rows;
    }
    for (int t=0;t<nthreads;++t) pthread_join(threads[t], NULL);
    free(threads); free(args);
}

void elemwise_add(real *X, real *Y, real *C, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    ElemArgs *args = malloc(nthreads * sizeof(ElemArgs));
    int base = n / nthreads, rem = n % nthreads, cur = 0;
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].row_start = cur;
        args[t].row_end = cur + rows;
        args[t].n = n;
        args[t].X = X; args[t].Y = Y; args[t].C = C;
        args[t].op = 2;
        pthread_create(&threads[t], NULL, elem_worker, &args[t]);
        cur += rows;
    }
    for (int t=0;t<nthreads;++t) pthread_join(threads[t], NULL);
    free(threads); free(args);
}

void scale_parallel(real *X, real *C, real alpha, int n, int nthreads) {
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    ElemArgs *args = malloc(nthreads * sizeof(ElemArgs));
    int base = n / nthreads, rem = n % nthreads, cur = 0;
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].row_start = cur;
        args[t].row_end = cur + rows;
        args[t].n = n;
        args[t].X = X; args[t].C = C;
        args[t].alpha = alpha;
        args[t].op = 3;
        pthread_create(&threads[t], NULL, elem_worker, &args[t]);
        cur += rows;
    }
    for (int t=0;t<nthreads;++t) pthread_join(threads[t], NULL);
    free(threads); free(args);
}

// compute max column sum (1-norm) and max row sum (inf-norm)
real norm_one(real *A, int n) {
    real maxc = 0.0;
    for (int j = 0; j < n; ++j) {
        real s = 0.0;
        for (int i = 0; i < n; ++i) s += fabs(A[i*n + j]);
        if (s > maxc) maxc = s;
    }
    return maxc;
}
real norm_inf(real *A, int n) {
    real maxr = 0.0;
    for (int i = 0; i < n; ++i) {
        real s = 0.0;
        for (int j = 0; j < n; ++j) s += fabs(A[i*n + j]);
        if (s > maxr) maxr = s;
    }
    return maxr;
}

// Frobenius norm of I - A*X
real residual_frobenius(real *A, real *X, int n, int nthreads) {
    // compute AX into tmp
    real *tmp = (real*)calloc((size_t)n*(size_t)n, sizeof(real));
    matmul_parallel(A, X, tmp, n, nthreads);
    real s = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            real val = tmp[i*n + j] - (i==j ? 1.0 : 0.0);
            s += val*val;
        }
    }
    free(tmp);
    return sqrt(s);
}

// transpose into B
void transpose(real *A, real *B, int n) {
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) B[j*n + i] = A[i*n + j];
}

// Newton-Schulz iteration:
// X0 provided, output in X (overwritten)
int newton_schulz_inverse(real *A, real *X, int n, int nthreads, int max_iters, double eps) {
    // using temporaries: R (n*n), tmp (n*n), AX (n*n)
    real *AX = (real*)malloc((size_t)n*(size_t)n*sizeof(real));
    real *R  = (real*)malloc((size_t)n*(size_t)n*sizeof(real));
    real *Xnew = (real*)malloc((size_t)n*(size_t)n*sizeof(real));

    for (int iter = 0; iter < max_iters; ++iter) {
        // AX = A * X
        matmul_parallel(A, X, AX, n, nthreads);

        // R = 2I - AX   --> R = (-1)*AX, then add 2I
        // reuse R: R = -AX
        scale_parallel(AX, R, -1.0, n, nthreads);
        // add 2I: R[i,i] += 2.0
        for (int i = 0; i < n; ++i) R[i*n + i] += 2.0;

        // Xnew = X * R
        matmul_parallel(X, R, Xnew, n, nthreads);

        // compute residual ||I - A Xnew||_F
        real res = residual_frobenius(A, Xnew, n, nthreads);
        // swap pointers X <-> Xnew
        memcpy(X, Xnew, (size_t)n*(size_t)n*sizeof(real));

        if (res < eps) {
            free(AX); free(R); free(Xnew);
            return iter+1; // converged in iter+1 iterations
        }
    }
    free(AX); free(R); free(Xnew);
    return -1; // not converged
}

// read matrix from file
int read_matrix_from_file(const char *filename, real **A_out, int *n_out) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Could not open input file\n");
        return -1;
    }
    if (fscanf(f, "%d", n_out) != 1) {
        fprintf(stderr, "Error reading matrix size\n");
        fclose(f);
        return -1;
    }
    int n = *n_out;
    *A_out = (real*)malloc((size_t)n * n * sizeof(real));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fscanf(f, "%lf", &(*A_out)[i*n + j]) != 1) {
                fprintf(stderr, "Error reading matrix element [%d,%d]\n", i, j);
                fclose(f);
                free(*A_out);
                return -1;
            }
        }
    }
    fclose(f);
    return 0;
}

// write matrix into file
int write_matrix_to_file(const char *filename, real *A, int n) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Could not open output file\n");
        return -1;
    }
    fprintf(f, "%.10d\n", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fprintf(f, "%.10f ", A[i*n + j]);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void print_usage(char *p) {
    fprintf(stderr, "Usage: %s input.txt output.txt [threads max_iters eps]\n", p);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int nthreads = (argc >= 4) ? atoi(argv[3]) : 4;
    int max_iters = (argc >= 5) ? atoi(argv[4]) : 50;
    double eps = (argc >= 6) ? atof(argv[5]) : 1e-8;

    real *A = NULL;
    int n;
    if (read_matrix_from_file(input_file, &A, &n) != 0) {
        return 1;
    }

    fprintf(stderr, "Reading %dx%d matrix from '%s'\n", n, n, input_file);

    real *At = (real*)malloc((size_t)n*n*sizeof(real));
    real *X = (real*)malloc((size_t)n*n*sizeof(real));
    transpose(A, At, n);

    real n1 = norm_one(A, n);
    real ninf = norm_inf(A, n);
    real scale = n1 * ninf;
    if (scale == 0.0) scale = 1.0;

    for (size_t i=0; i<(size_t)n*n; ++i) X[i] = At[i] / scale;

    double t0 = now_seconds();
    int iters = newton_schulz_inverse(A, X, n, nthreads, max_iters, eps);
    double t1 = now_seconds();

    if (iters > 0)
        fprintf(stderr, "Converged in %d iterations, time %.4f s\n", iters, t1 - t0);
    else
        fprintf(stderr, "Not converged in %d iterations, time %.4f s\n", max_iters, t1 - t0);

    real res = residual_frobenius(A, X, n, nthreads);
    fprintf(stderr, "Residual Frobenius ||I - AX||_F = %.6e\n", res);

    if (write_matrix_to_file(output_file, X, n) == 0)
        fprintf(stderr, "Result saved in '%s'\n", output_file);

    free(A); free(At); free(X);
    return 0;
}
