#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;
long long monteCarloPi(long long num_samples)
{
    long long inside_circle = 0;
    for (long long i = 0; i < num_samples; i++)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            inside_circle++;
        }
    }
    return inside_circle;
}

int main(int argc, char **argv)
{
    int rank, size;
    long long total_samples = 10000000;
    long long local_samples, local_count, global_count;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank);
    local_samples = total_samples / size;
    local_count = monteCarloPi(local_samples);
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double pi_estimate = (4.0 * global_count) / total_samples;
        cout << "Estimated value of Pi: " << pi_estimate << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
