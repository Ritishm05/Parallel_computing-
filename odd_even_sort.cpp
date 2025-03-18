#include <stdio.h>
#include<iostream>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

void odd_even_sort(int *local_array, int local_n, int world_size, int world_rank) {
    int phase, temp;
    MPI_Status status;
    
    for (phase = 0; phase < world_size; phase++) {
        if (phase % 2 == 0) {  // Even phase
            if (world_rank % 2 == 0) {
                if (world_rank + 1 < world_size) {
                    MPI_Sendrecv(&local_array[local_n - 1], 1, MPI_INT, world_rank + 1, 0,
                                 &temp, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, &status);
                    if (temp < local_array[local_n - 1]) {
                        local_array[local_n - 1] = temp;
                    }
                }
            } else {
                MPI_Sendrecv(&local_array[0], 1, MPI_INT, world_rank - 1, 0,
                             &temp, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, &status);
                if (temp > local_array[0]) {
                    local_array[0] = temp;
                }
            }
        } else { 
            if (world_rank % 2 != 0) {
                if (world_rank + 1 < world_size) {
                    MPI_Sendrecv(&local_array[local_n - 1], 1, MPI_INT, world_rank + 1, 0,
                                 &temp, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, &status);
                    if (temp < local_array[local_n - 1]) {
                        local_array[local_n - 1] = temp;
                    }
                }
            } else if (world_rank > 0) {
                MPI_Sendrecv(&local_array[0], 1, MPI_INT, world_rank - 1, 0,
                             &temp, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, &status);
                if (temp > local_array[0]) {
                    local_array[0] = temp;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int world_size, world_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int local_n = 1; 
    int local_array[1];
    
    if (world_rank == 0) {
        int n;
        cin>>n;
        int global_array[n];
        for(int i=0;i<n;i++){
            cin>>global_array[i];
        }

        int n = sizeof(global_array) / sizeof(global_array[0]);
        for (int i = 0; i < world_size; i++) {
            MPI_Send(&global_array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Recv(&local_array[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    odd_even_sort(local_array, local_n, world_size, world_rank);
    
    MPI_Gather(local_array, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("Sorted array: ");
        for (int i = 0; i < world_size; i++) {
            int value;
            MPI_Recv(&value, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%d ", value);
        }
        printf("\n");
    }
    else {
        MPI_Send(&local_array[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}