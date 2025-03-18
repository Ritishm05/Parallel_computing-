#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 4;
    const int local_rows = N / world_size;
    std::vector<int> local_matrix(local_rows * N);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_matrix[i * N + j] = (world_rank * local_rows + i) * N + j; 
        }
    }

    std::cout << "Rank " << world_rank << " original block:\n";
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(3) << local_matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::vector<int> transposed_block(local_rows * N);

    MPI_Datatype block_type;
    MPI_Type_vector(local_rows, 1, N, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    MPI_Alltoall(
        local_matrix.data(),       
        1, block_type,             
        transposed_block.data(),   
        local_rows * N / world_size, MPI_INT, 
        MPI_COMM_WORLD
    );

    MPI_Type_free(&block_type);
    std::cout << "Rank " << world_rank << " transposed block:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < local_rows; j++) {
            std::cout << std::setw(3) << transposed_block[j + i * local_rows] << " ";
        }
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
