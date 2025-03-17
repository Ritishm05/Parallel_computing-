#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<MPI_Wtime>
#define N 70  

void sequential_matrix_multiplication(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void initialize_matrix(double M[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = rand() % 10;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[N][N], B[N][N], C[N][N];
    double start_time, run_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        initialize_matrix(A);
        initialize_matrix(B);
    }

    if (rank == 0) {
        start_time = MPI_W.time();
        sequential_matrix_multiplication(A, B, C);
        run_time = MPI_Wtime() - start_time;
        printf("Sequential Execution Time: %f seconds\n", run_time);
    }

    int rows_per_process = N / size; 
    double local_A[rows_per_process][N], local_C[rows_per_process][N];

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start_time = MPI_Wtime();

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }
    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        run_time = MPI_Wtime() - start_time;
        printf("Parallel Execution Time with %d processes: %f seconds\n", size, run_time);
    }

    MPI_Finalize();
    return 0;
}
