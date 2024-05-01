#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h> // include string.h for strchr

#define TILE_SIZE (matrix_size / p)  // tile size

// function for printing a matrix
void print_matrix(double *matrix, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%6.2f ", matrix[i * matrix_size + j]);
        }
        printf("\n");
    }
}

// function for allocationg a square matrix, double precision
double* allocate_matrix(int dim) {
    // allocate a block of memory for our matrix, returns a pointer.
    double* matrix = (double*)malloc(dim * dim * sizeof(double));
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

// function for distributing blocks of matrices to the different processes
void distribute_blocks(double** A, double** B, int matrix_size, int processes, int rank, int block_size, MPI_Comm grid_comm) {
    // create a new datatype for the blocks
    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, matrix_size, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    // distribute the blocks of matrix A
    for (int i = 0; i < processes; i++) {
        if (rank == 0) {
            // send a block of matrix A to process i
            MPI_Send(&(*A)[i * block_size * block_size], 1, block_type, i, 0, grid_comm);
        } else if (rank == i) {
            // receive a block of matrix A from the master process
            MPI_Recv(*A, block_size * block_size, MPI_DOUBLE, 0, 0, grid_comm, MPI_STATUS_IGNORE);
        }
    }

    // distribute the blocks of matrix B
    for (int i = 0; i < processes; i++) {
        if (rank == 0) {
            // send a block of matrix A to process i
            MPI_Send(&(*B)[i * block_size * block_size], 1, block_type, i, 0, grid_comm);
        } else if (rank == i) {
            // receive a block of matrix A from the master process
            MPI_Recv(*B, block_size * block_size, MPI_DOUBLE, 0, 0, grid_comm, MPI_STATUS_IGNORE);
        }
    }

    // free the datatype when it is no longer needed
    MPI_Type_free(&block_type);
}

//function to gather the result matrix from all processes and assemble the full matrix on the master process
void gather_results(double *C, double *C_full, int tile_size, MPI_Comm grid_comm) {
    // gather all blocks of C from each process
    MPI_Gather(C, tile_size * tile_size, MPI_DOUBLE, C_full, tile_size * tile_size, MPI_DOUBLE, 0, grid_comm);
}

// function for reading matrices A and B from input file
void read_input_matrices_from_file(const char* filename, double** A, double** B, int* matrix_size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("[ERROR] No path to matrix file.\n");
        exit(EXIT_FAILURE);
    }

    // read the matrix size from the first line of the file
    fscanf(file, "%d", matrix_size);
    // allocate matrix A, B
    *A = allocate_matrix(*matrix_size);
    *B = allocate_matrix(*matrix_size);

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

// function to read a single matrix from a file, used in the test comparison
void read_expected_matrix_from_file(const char* filename, double** matrix, int matrix_size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("[ERROR] No path to matrix file.\n");
        exit(EXIT_FAILURE);
    }
    
    // allocate for matrix
    *matrix = allocate_matrix(matrix_size);

    char line[1024]; // buffer to read each line

    // loop through the file to find the line with 'E'
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, 'E') != NULL) {
            break; // stop when we find a line containing 'E'
        }
    }
    // start reading the matrix
    for (int i = 0; i < (matrix_size); i++) {
        for (int j = 0; j < (matrix_size); j++) {
            fscanf(file, "%lf", &(*matrix)[i * matrix_size + j]);
        }
    }
    
    fclose(file);
}

// function to compare two matrices
bool compare_matrices(double* calculated_matrix, double* expected_matrix, int matrix_size, double tolerance) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            // compare the values with a tolerance
            if (fabs(calculated_matrix[i * matrix_size + j] - expected_matrix[i * matrix_size + j]) > tolerance) {
                return false; // the matrices are not the same
            }
        }
    }
    return true; // the matrices are the same
}

void test_matrix_corectness(double* calculated_matrix, int matrix_size, char* file_path) {
    double* expected_matrix;

    // read expected matrix, pass as reference
    read_expected_matrix_from_file(file_path, &expected_matrix, matrix_size);
    // set tolerance level
    double tolerance = 1e-6;
    bool matrices_match = compare_matrices(calculated_matrix, expected_matrix, matrix_size, tolerance);
    
    // check if test passes
    if (matrices_match) {
        printf("[TEST PASS] The calculated matrix matches the expected matrix.\n");
    } else {
        printf("[TEST FAIL] The calculated matrix does not match the expected matrix.\n");
    }

    // free up memory use
    free(expected_matrix);
}

int main(int argc, char** argv) {
    int processes, rank;
    int p;                         // grid size (square root of num of processes)
    int grid_rank, grid_coords[2]; // cartesian grid
    MPI_Comm grid_comm;            // communicator with cartesian topology
    char* input_file = NULL;       // path to file

    // initialize MPI environment
    MPI_Init(&argc, &argv);
    // get the rank ID of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // print the hello message for each MPI process
    printf("Hello from rank %d from %d processes!\n", rank, processes);

    // check if input file name is provided as an argument
    if (argc > 1) {
        input_file = argv[1]; // the file name is the first argument after the program name
    } else {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // set grid size
    p = (int)sqrt(processes);
    // special case if processes = 2, ALWAYS CRASH FOR SOME REASON
    if (p * p != processes) {
        // print on master process only
        if (rank == 0) {
            printf("[ERROR] The number of processes must be a integer square, is %i.", p);
        }
        // quit
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // initilize the blocks
    double *A, *B;
    int matrix_size;

    // read input matrices,  pass as reference, only on master process
    if (rank == 0) {
        read_input_matrices_from_file(input_file, &A, &B, &matrix_size);

        // check matrix size
        if (matrix_size % p != 0) {
            printf("[ERROR] The matrix size must be divisible by the root of the number of processes.");

            // quit
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    // broadcast matrix_size to all processes, it is needed in fox algorithm
    //MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // create cartesian grid, set variables
    int ndims = 2;                  // number of dimensions in grid, always 2D
    int dims[2] = {p, p};           // integer array of size ndims, specifying number of processes in each dimension
    int periods[2] = {1, 1};        // "boolean" array, use periodic boundaries (wrap around) for both dimensions
    int reorder = 1;                // "boolean", let MPI reorder ranks
    // initilize grid, all processes need the grid to know the coordinates of the blocks
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, ndims, grid_coords);

    // allocate C matrix
    double* C = allocate_matrix(TILE_SIZE);
    // allocate full matrices (on rank 0)
    double* C_full = NULL;
    if (rank == 0) {
        // use whole matrix size since it is the full matrix
        C_full = allocate_matrix(matrix_size);
    }

    // distribute the blocks
    //distribute_blocks(&A, &B, matrix_size, processes, rank, TILE_SIZE, grid_comm);

    // run fox algorithm
    for (int step = 0; step < p; step++) {
        // set rank of source process for block A
        int source_A;
        MPI_Cart_shift(grid_comm, 1, -step, &grid_rank, &source_A);

        // send and recieve blocks of A
        MPI_Sendrecv_replace(A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, source_A, 0, source_A, 0, grid_comm, MPI_STATUS_IGNORE);
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
        print_matrix(C_full, matrix_size);
        // compare
        test_matrix_corectness(C_full, matrix_size, input_file);
    }

    // clean up memory allocations
    free(A);
    free(B);
    free(C);
    //free(A_shift);
    //free(B_shift);
    // free upp full matrix if master process
    if (rank == 0) {
        free(C_full);
    }
 
    // finalize the MPI environment
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}