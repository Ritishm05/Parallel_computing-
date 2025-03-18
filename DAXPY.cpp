#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <cmath>     
#include <iomanip>   

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = 65536;
    double a = 2.5;
    std::vector<double> X_full, Y_full;
    int local_N = N / world_size;
    std::vector<double> X_local(local_N);
    std::vector<double> Y_local(local_N);
    if (world_rank == 0) {
        X_full.resize(N);
        Y_full.resize(N);
        for (int i = 0; i < N; ++i) {
            X_full[i] = rand() % 100;
            Y_full[i] = rand() % 100;
        }
    }
    MPI_Scatter(X_full.data(), local_N, MPI_DOUBLE, X_local.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y_full.data(), local_N, MPI_DOUBLE, Y_local.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // SERIAL IMPLEMENTATION 
    double serial_start = MPI_Wtime();
    if (world_rank == 0) {
        for (int i = 0; i < N; ++i) {
            X_full[i] = a * X_full[i] + Y_full[i];
        }
    }
    double serial_end = MPI_Wtime();

    //  PARALLEL IMPLEMENTATION
    double parallel_start = MPI_Wtime();
    for (int i = 0; i < local_N; ++i) {
        X_local[i] = a * X_local[i] + Y_local[i];
    }
    MPI_Gather(X_local.data(), local_N, MPI_DOUBLE, X_full.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double parallel_end = MPI_Wtime();

    //OUTPUT
    if (world_rank == 0) {
        double serial_time = serial_end - serial_start;
        double parallel_time = parallel_end - parallel_start;
        double speedup = serial_time / parallel_time;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Serial time   : " << serial_time << " seconds\n";
        std::cout << "Parallel time : " << parallel_time << " seconds\n";
        std::cout << "Speedup       : " << speedup << "x" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
