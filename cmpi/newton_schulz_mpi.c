// newton_schulz_mpi.c
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

typedef double real;

static inline size_t idx(size_t n, size_t i, size_t j) { return i*n + j; }

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int read_matrix_from_file(const char *filename, real **A_out, int *n_out) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Could not open input file");
        return -1;
    }
    if (fscanf(f, "%d", n_out) != 1) {
        fprintf(stderr, "Error reading matrix size\n");
        fclose(f);
        return -1;
    }
    int n = *n_out;
    *A_out = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    if (!*A_out) { fclose(f); return -1; }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fscanf(f, "%lf", &(*A_out)[idx((size_t)n,(size_t)i,(size_t)j)]) != 1) {
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

int write_matrix_to_file(const char *filename, real *A, int n) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Could not open output file");
        return -1;
    }
    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fprintf(f, "%.18f%c", A[idx((size_t)n,(size_t)i,(size_t)j)], (j+1==n?'\n':' '));
    }
    fclose(f);
    return 0;
}

// partitioning rows among ranks
void compute_counts_displs(int n, int size, int *counts_rows, int *displs_rows) {
    int base = n / size;
    int rem = n % size;
    // fprintf(stderr, "n = %d, size = %d, base = %d, rem = %d\n", n, size, base, rem);
    int cur = 0;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        counts_rows[r] = rows;
        displs_rows[r] = cur;
        cur += rows;
        // fprintf(stderr, "r = %d, counts_rows[r] = %d, displs_rows[r] = %d, cur = %d\n", r, counts_rows[r], displs_rows[r], cur);
    }
}

// Distributed matrix multiply: C = A * B
int matmul_distributed(const real *A, const real *B, real *C_out, int n,
                       const int *counts_rows, const int *displs_rows, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int rows_local = counts_rows[rank];
    if (rows_local == 0) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C_out, NULL, NULL, MPI_DOUBLE, comm);
        return 0;
    }
    real *local_C = (real*)calloc((size_t)rows_local * (size_t)n, sizeof(real));
    if (!local_C) return -1;

    int rstart = displs_rows[rank];
    for (int i = 0; i < rows_local; ++i) {
        const real *Ai = &A[(size_t)(rstart + i) * (size_t)n];
        real *Ci = &local_C[(size_t)i * (size_t)n];
        // initialize row
        for (int j = 0; j < n; ++j) Ci[j] = 0.0;
        // standard ijk with A row, B rows
        for (int k = 0; k < n; ++k) {
            real a = Ai[k];
            const real *Bk = &B[(size_t)k * (size_t)n];
            for (int j = 0; j < n; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }

    // prepare recvcounts/displs for Allgatherv in elements (doubles)
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *recvdispls = (int*)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        recvcounts[r] = counts_rows[r] * n;
        recvdispls[r] = displs_rows[r] * n;
    }

    MPI_Allgatherv(local_C, rows_local * n, MPI_DOUBLE,
                   C_out, recvcounts, recvdispls, MPI_DOUBLE, comm);

    free(local_C);
    free(recvcounts);
    free(recvdispls);
    return 0;
}

// Distributed scale: C = alpha * X
int scale_distributed(const real *X, real *C_out, real alpha, int n,
                      const int *counts_rows, const int *displs_rows, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int rows_local = counts_rows[rank];
    real *local_C = NULL;
    if (rows_local > 0) {
        local_C = (real*)malloc((size_t)rows_local * (size_t)n * sizeof(real));
        int rstart = displs_rows[rank];
        for (int i = 0; i < rows_local; ++i) {
            const real *Xi = &X[(size_t)(rstart + i) * (size_t)n];
            real *Ci = &local_C[(size_t)i * (size_t)n];
            for (int j = 0; j < n; ++j) Ci[j] = alpha * Xi[j];
        }
    }

    int *recvcounts = (int*)malloc(sizeof(int) * size);
    int *recvdispls = (int*)malloc(sizeof(int) * size);
    for (int r = 0; r < size; ++r) {
        recvcounts[r] = counts_rows[r] * n;
        recvdispls[r] = displs_rows[r] * n;
    }

    MPI_Allgatherv(local_C, rows_local * n, MPI_DOUBLE,
                   C_out, recvcounts, recvdispls, MPI_DOUBLE, comm);

    if (local_C) free(local_C);
    free(recvcounts);
    free(recvdispls);
    return 0;
}

double residual_frobenius_distributed(const real *A, const real *X, int n,
                                      const int *counts_rows, const int *displs_rows, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rows_local = counts_rows[rank];
    int rstart = displs_rows[rank];

    // compute local AX rows
    double local_sum = 0.0;
    for (int i = 0; i < rows_local; ++i) {
        int gi = rstart + i;
        const real *Ai = &A[(size_t)gi * (size_t)n];
        // compute row gi of AX
        for (int j = 0; j < n; ++j) {
            double val = 0.0;
            const real *Bj = &X[(size_t)j * (size_t)n]; // X row j
            // but X is row-major: X[j*n + k] gives row j; we need column? simpler do dot product:
            // val = sum_k A[gi,k] * X[k,j]
            for (int k = 0; k < n; ++k) {
                val += Ai[k] * X[(size_t)k * (size_t)n + (size_t)j];
            }
            double diff = val - ((gi == j) ? 1.0 : 0.0);
            local_sum += diff * diff;
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sqrt(global_sum);
}

// transpose (only rank 0)
void transpose_inplace(const real *A, real *B, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[(size_t)j * (size_t)n + (size_t)i] = A[(size_t)i * (size_t)n + (size_t)j];
}

// compute 1-norm and inf-norm on rank 0
real norm_one(const real *A, int n) {
    real maxc = 0.0;
    for (int j = 0; j < n; ++j) {
        real s = 0.0;
        for (int i = 0; i < n; ++i) s += fabs(A[(size_t)i * (size_t)n + (size_t)j]);
        if (s > maxc) maxc = s;
    }
    return maxc;
}

real norm_inf(const real *A, int n) {
    real maxr = 0.0;
    for (int i = 0; i < n; ++i) {
        real s = 0.0;
        for (int j = 0; j < n; ++j) s += fabs(A[(size_t)i * (size_t)n + (size_t)j]);
        if (s > maxr) maxr = s;
    }
    return maxr;
}

int newton_schulz_inverse_distributed(real *A, real *X, int n,
                                      const int *counts_rows, const int *displs_rows,
                                      MPI_Comm comm, int max_iters, double eps,
                                      int rank0_print) {
    // temporaries
    real *AX = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    real *R  = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    real *Xnew = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    if (!AX || !R || !Xnew) {
        free(AX); free(R); free(Xnew);
        return -1;
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // AX = A * X  (distributed)
        if (matmul_distributed(A, X, AX, n, counts_rows, displs_rows, comm) != 0) {
            if (rank0_print) fprintf(stderr, "matmul_distributed failed\n");
            free(AX); free(R); free(Xnew);
            return -1;
        }

        // R = -AX  then add 2I
        if (scale_distributed(AX, R, -1.0, n, counts_rows, displs_rows, comm) != 0) {
            if (rank0_print) fprintf(stderr, "scale_distributed failed\n");
            free(AX); free(R); free(Xnew);
            return -1;
        }

        for (int i = 0; i < n; ++i) R[(size_t)i * (size_t)n + (size_t)i] += 2.0;

        // Xnew = X * R
        if (matmul_distributed(X, R, Xnew, n, counts_rows, displs_rows, comm) != 0) {
            if (rank0_print) fprintf(stderr, "matmul_distributed failed\n");
            free(AX); free(R); free(Xnew);
            return -1;
        }

        // compute residual ||I - A Xnew||_F (distributed)
        double res = residual_frobenius_distributed(A, Xnew, n, counts_rows, displs_rows, comm);

        // copy Xnew -> X (full)
        memcpy(X, Xnew, (size_t)n * (size_t)n * sizeof(real));

        int converged = (res < eps);
        if (rank0_print) {
            // only rank 0 prints outer info
            int global_rank;
            MPI_Comm_rank(comm, &global_rank);
            if (global_rank == 0) {
                fprintf(stderr, "iter %3d: residual = %.6e%s\n", iter+1, res, converged ? " (converged)" : "");
            }
        }
        if (converged) {
            free(AX); free(R); free(Xnew);
            return iter+1;
        }
    }

    free(AX); free(R); free(Xnew);
    return -1;
}

void print_usage(char *p) {
    fprintf(stderr, "Usage: %s input.txt output.txt [max_iters eps]\nMPI processes set by mpirun/mpiexec.\n", p);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int max_iters = (argc >= 4) ? atoi(argv[3]) : 50;
    double eps = (argc >= 5) ? atof(argv[4]) : 1e-8;

    real *A = NULL;
    int n = 0;

    // only rank0 reads
    if (rank == 0) {
        if (read_matrix_from_file(input_file, &A, &n) != 0) {
            fprintf(stderr, "Rank 0: failed to read input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(stderr, "Rank 0: Read %dx%d matrix from '%s'\n", n, n, input_file);
    }

    // broadcast n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size n=%d\n", n);
        MPI_Finalize();
        return 1;
    }

    // allocate A on all ranks and bcast contents. all processes store matrices at least for now. todo: bcast only respective rows to each process?
    if (rank != 0) A = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // compute partitioning of rows
    int *counts_rows = (int*)malloc(size * sizeof(int));
    int *displs_rows = (int*)malloc(size * sizeof(int));
    compute_counts_displs(n, size, counts_rows, displs_rows);


    // allocate matrices present on all ranks
    real *At = (real*)malloc((size_t)n * (size_t)n * sizeof(real));
    real *X = (real*)malloc((size_t)n * (size_t)n * sizeof(real));

    // rank 0 computes At and initial X, others wait and receive via Bcast
    if (rank == 0) {
        transpose_inplace(A, At, n);
        real n1 = norm_one(A, n);
        real ninf = norm_inf(A, n);
        real scale = n1 * ninf;
        if (scale == 0.0) scale = 1.0;
        for (size_t i = 0; i < (size_t)n * (size_t)n; ++i) X[i] = At[i] / scale;
    }

    MPI_Bcast(At, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(X, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stderr, "Starting Newton-Schulz with %d MPI ranks, max_iters=%d, eps=%.3e\n", size, max_iters, eps);
    }

    double t0 = MPI_Wtime();
    int iters = newton_schulz_inverse_distributed(A, X, n, counts_rows, displs_rows, MPI_COMM_WORLD, max_iters, eps, 1);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        if (iters > 0)
            fprintf(stderr, "Converged in %d iterations, time %.4f s\n", iters, t1 - t0);
        else
            fprintf(stderr, "Not converged in %d iterations, time %.4f s\n", max_iters, t1 - t0);
    }

    // compute final residual and print on rank 0
    double res = residual_frobenius_distributed(A, X, n, counts_rows, displs_rows, MPI_COMM_WORLD);
    if (rank == 0) fprintf(stderr, "Residual Frobenius ||I - AX||_F = %.6e\n", res);

    // write result: only rank 0
    if (rank == 0) {
        if (write_matrix_to_file(output_file, X, n) == 0)
            fprintf(stderr, "Result saved in '%s'\n", output_file);
    }

    free(A); free(At); free(X);
    free(counts_rows); free(displs_rows);

    MPI_Finalize();
    return 0;
}
