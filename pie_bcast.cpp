#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    static long num_steps = 1e8; 
    double step;
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;
    double x, local_sum = 0.0;
    for (long i = world_rank; i < num_steps; i += world_size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double pi = step * global_sum;
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Approximated value of pi = " << pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
