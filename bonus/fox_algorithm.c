#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// function for allocationg and initializing a matrix, double precision
double*  allocate_matrix(int rows, int cols, int fill_matrix) {
    // allocate a block of memory for our matrix, returns a pointer.
    double* matrix = (double*)malloc(rows * cols * sizeof(double));

    // if fill matrix is true, fill with random values
    if (fill_matrix) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = (double)rand() / RAND_MAX;
        }
    }

    return matrix;
}

int main(int argc, char** argv) {
    int processes, rank;
    int p;                         // grid size (square root of num of processes)
    int grid_rank, grid_coords[2]; // cartesian grid
    MIP_Comm grid_comm;            // communicator with cartesian topology

    // initialize MPI environment
    MPI_Init(&argc, &argv);
    // get the rank ID of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // set grid size
    p = (int)sqrt(processes);
    if (p * p != processes) {
        // print on one process only
        if (rank == 0) {
            printf("[ERROR] The number of processes must be a integer square.")
        }
        // quit
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // create cartesian grid, set variables
    int ndims = 2;                  // number of dimensions in grid, always 2D
    int dims[ndims] = {p, p};       // integer array of size ndims, specifying number of processes in each dimension
    int periods[ndims] = {1, 1};    // "boolean" array, use periodic boundaries (wrap around) for both dimensions
    int reorder = 1;                // "boolean", let MPI reorder ranks
    // initilize grid
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, ndims, grid_coords);

    // allocate matrix A, B, C
    double* A = allocate_matrix(1, 1, 1);
    double* B = allocate_matrix(1, 1, 1);
    double* C = allocate_matrix(1, 1, 0);

    // clean up memory allocations
    free(A);
    free(B);
    free(C);

    // finalize the MPI environment
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}