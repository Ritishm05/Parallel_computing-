#include <mpi.h>
#include <iostream>
#include <cmath>
using namespace std;

bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <=sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int MASTER = 0;
    const int MAX_VALUE = 100;

    if (world_rank == MASTER) {
        // MASTER PROCESS
        int current = 2; 
        int active_slaves = world_size - 1;

        while (active_slaves > 0) {
            int msg;
            MPI_Status status;
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int slave_rank = status.MPI_SOURCE;

            if (msg == 0) {
            } else if (msg > 0) {
                cout << "Master received: " << msg << " is PRIME from slave " << slave_rank << std::endl;
            } else if (msg < 0) {
                cout << "Master received: " << -msg << " is NOT prime from slave " << slave_rank << std::endl;
            }
            if (current <= MAX_VALUE) {
                MPI_Send(&current, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
                current++;
            } else {
                int stop = -1;
                MPI_Send(&stop, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
                active_slaves--;
            }
        }

    } else {
        //SLAVE PROCESSES
        int request = 0; 
        MPI_Send(&request, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

        while (true) {
            int n;
            MPI_Recv(&n, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (n == -1) {
                break;
            }

            if (is_prime(n)) {
                MPI_Send(&n, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            } else {
                int non_prime = -n;
                MPI_Send(&non_prime, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
