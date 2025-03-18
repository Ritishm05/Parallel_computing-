#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int local_value = world_rank + 1;

    int prefix_sum = 0;
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "Rank " << world_rank << ": prefix sum = " << prefix_sum << std::endl;

    MPI_Finalize();
    return 0;
}
