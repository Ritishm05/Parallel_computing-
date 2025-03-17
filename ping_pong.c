#include <mpi.h>
#include <stdio.h>

#define PING_PONG_LIMIT 10

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires at least 2 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ping_pong_count = 0;
    int partner_rank = (rank == 0) ? 1 : 0;

    while (ping_pong_count < PING_PONG_LIMIT) {
        if (rank == ping_pong_count % 2) {
            ping_pong_count++;
            MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            printf("Process %d sent ping_pong_count %d to process %d\n", rank, ping_pong_count, partner_rank);
        } else {
            MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d received ping_pong_count %d from process %d\n", rank, ping_pong_count, partner_rank);
        }
    }

    MPI_Finalize();
    return 0;
}
