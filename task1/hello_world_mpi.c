#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int processes, rank;

    // initialize MPI environment
    MPI_Init(&argc, &argv);

    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // get the rank ID of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // print the hello world message for each MPI process
    printf("Hello World from rank %d from %d processes!\n", rank, processes);

    // finalize the MPI environment
    MPI_Finalize();

    return 0;
}