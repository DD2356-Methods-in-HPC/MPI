#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE (matrix_size / p)

// function for allocationg a square matrix, double precision
double* allocate_matrix(int dim) {
    // allocate a block of memory for our matrix, returns a pointer.
    double* matrix = (double*)malloc(dim * dim * sizeof(double));

    /*
    // if fill matrix is true, fill with random values
    if (fill_matrix) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = (double)rand() / RAND_MAX;
        }
    }
    */
    return matrix;
}

// function for matrix multiplication, two blocks (A & B) => one block (C)
void multiply_accumalate(double* A, double* B, double* C, int size) {
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < size; z++) {
                C[x * size + y] += A[x * size + z] * B[z * size + y];
            }
        }
    }
}

// TEMPORARY
void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

//function to gather the result matrix from all processes and assemble the full matrix on the master process
void gather_results(double *C, double *C_full, int TILE_SIZE, MPI_Comm grid_comm) {
    // gather all blocks of C from each process
    MPI_Gather(C, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, C_full, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, 0, grid_comm);
}

// function for reading matrices from input file
void read_matrices_from_file(const char* filename, double** A, double** B, int* matrix_size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(EXIT_FAILURE);
    }

    // read the matrix size from the first line of the file
    fscanf(file, "%d", matrix_size);
    // allocate matrix A, B
    *A = allocate_matrix(*matrix_size); //(double*)malloc((*matrix_size) * (*matrix_size) * sizeof(double));
    *B = allocate_matrix(*matrix_size); //(double*)malloc((*matrix_size) * (*matrix_size) * sizeof(double));

    // read matrix A from the second line
    for (int i = 0; i < (*matrix_size) * (*matrix_size); i++) {
        fscanf(file, "%lf", &(*A)[i]);
    }

    // read matrix B from the third line
    for (int i = 0; i < (*matrix_size) * (*matrix_size); i++) {
        fscanf(file, "%lf", &(*B)[i]);
    }

    // close the input file
    fclose(file);
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

    // initilize the blocks
    double *A, *B;
    int matrix_size;

    // read matrix values,  pass as reference
    read_matrices_from_file("test.txt", &A, &B, &matrix_size);

    // allocate C matrix
    double* C = allocate_matrix(TILE_SIZE);
    // allocate buffers for shifting blocks
    double* A_shift = allocate_matrix(TILE_SIZE);
    double* B_shift = allocate_matrix(TILE_SIZE);

    // allocate full matrices (on rank 0)
    double* C_full = NULL;
    if (rank == 0) {
        // use whole matrix size since it is the full matrix
        C_full = allocate_matrix(matrix_size);
    }

    // run fox algorithm
    for (int step = 0; step < p; step++) {
        // set rank of source process for block A
        int source_A;
        MPI_Cart_shift(grid_comm, 1, -step, &grid_rank, &source_A);

        // send and recieve blocks of A
        MPI_Sendrecv_replace(A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, source_A, 0, source_A, 0, grid_comm; MPI_STATUS_IGNORE);
        // multiply blocks and accumulate result into C
        multiply_accumalate(A, B, C, TILE_SIZE);

        // shift block B left within each row
        int left, right;
        MPI_Cart_shift(grid_comm, 0, -1, &grid_rank, &right);
        MPI_Sendrecv_replace(B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, right, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // gather results to form the full matrix C on master process (rank 0)
    gather_results(C, C_full, TILE_SIZE, grid_comm);
    
    if (rank == 0) {
        // TODO: compare resulting matrix with answer?
        print_matrix(C_full, matrix_size, matrix_size);
    }

    // clean up memory allocations
    free(A);
    free(B);
    free(C);
    free(A_shift);
    free(B_shift);
    // free upp full matrix if master process
    if (rank == 0) {
        free(C_full);
    }
 
    // finalize the MPI environment
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}