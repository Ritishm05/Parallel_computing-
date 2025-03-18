#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 16; 
    int local_N = N / world_size;
    std::vector<double> local_vec_a(local_N);
    std::vector<double> local_vec_b(local_N);
    std::vector<double> vec_a, vec_b;
    if (world_rank == 0) {
        vec_a.resize(N);
        vec_b.resize(N);
        for (int i = 0; i < N; i++) {
            vec_a[i] = rand() % 10;
            vec_b[i] = rand() % 10;
        }
        std::cout << "Vector A: ";
        for (int i = 0; i < N; i++) std::cout << vec_a[i] << " ";
        std::cout << "\nVector B: ";
        for (int i = 0; i < N; i++) std::cout << vec_b[i] << " ";
        std::cout << std::endl;
    }

    MPI_Scatter(vec_a.data(), local_N, MPI_DOUBLE, local_vec_a.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vec_b.data(), local_N, MPI_DOUBLE, local_vec_b.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_dot = 0.0;
    for (int i = 0; i < local_N; i++) {
        local_dot += local_vec_a[i] * local_vec_b[i];
    }

    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Dot product result: " << global_dot << std::endl;
    }

    MPI_Finalize();
    return 0;
}
