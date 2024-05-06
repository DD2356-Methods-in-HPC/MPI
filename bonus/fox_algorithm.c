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
void distribute_blocks(double* A, double* B, double* local_A, double* local_B, int matrix_size, int rank, int processes, int block_size, MPI_Comm grid_comm) {
    // because the matrices are represented as a 1D array in the code (but we want them to function as 2D)
    // we have to be very careful with how we are defining our MPI datatype, especially with how the 
    // variables are stored in the memory.
    // To make sure our stride correctly reads the data and does not go out of bounds
    // we create a row_type first that defines the stride for each row,
    // then we can resize it for the blocks 
    MPI_Datatype row_type, block_type;

    // create a new data type for a row, it is the size of one dimension of a block
    MPI_Type_contiguous(block_size, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // create a new data type for a block, it is the size of one dimension of a block
    // but we set the stride as the original matrix size divided by the block size
    // we set row type as our original type
    MPI_Type_vector(block_size, 1, matrix_size/block_size, row_type, &block_type);
    // we then resize the block type with the size of a double to make sure that in memory space
    // we read correctly (not too far ahead)
    MPI_Type_create_resized(block_type, 0, sizeof(double) * block_size, &block_type);
    MPI_Type_commit(&block_type);

    // create arrays to hold the counts and displacements for MPI_Scatterv
    int* sendcounts = NULL;
    int* displacements = NULL;

    if (rank == 0) {
        int coords[2];
        sendcounts = malloc(processes * sizeof(int));
        displacements = malloc(processes * sizeof(int));

        for (int i = 0; i < processes; i++) {
            sendcounts[i] = 1; // sending one block of size block_size x block_size to each process

            // calculate the displacement for each process
            MPI_Cart_coords(grid_comm, i, 2, coords);

            displacements[i] = coords[0] * matrix_size + coords[1]; 
        }

        /*
        // debugging
        printf("\nDisplacements array:\n");
        for (int i = 0; i < processes; i++) {
                MPI_Cart_coords(grid_comm, i, 2, coords);
                printf("Process %d - coords (%d, %d), displacement: %d\n",
                    i, coords[0], coords[1], displacements[i]);
        }
        */
    }

    // before scatter, use barrier to synchronize processes
    MPI_Barrier(grid_comm);

    // scatter blocks of matrix A to all processes
    MPI_Scatterv(A, sendcounts, displacements, block_type, local_A, block_size * block_size, MPI_DOUBLE, 0, grid_comm);
    // scatter blocks of matrix B to all processes
    MPI_Scatterv(B, sendcounts, displacements, block_type, local_B, block_size * block_size, MPI_DOUBLE, 0, grid_comm);

    // after scatter, use barrier to synchronize processes
    MPI_Barrier(grid_comm);

    // free the arrays and datatype when they are no longer needed
    if (rank == 0) {
        free(sendcounts);
        free(displacements);
    }

    MPI_Type_free(&row_type);
    MPI_Type_free(&block_type);
}

// function to gather the result matrix from all processes and assemble the full matrix on the master process
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
        printf("\n[TEST PASS] The calculated matrix matches the expected matrix.\n");

    } else {
        printf("\n[TEST FAIL] The calculated matrix does not match the expected matrix.\nExpected matrix:\n");
        print_matrix(expected_matrix, matrix_size);
    }

    // free up memory use
    free(expected_matrix);
}

int main(int argc, char** argv) {
    int processes, rank;
    int p;                         // grid size (square root of num of processes)
    int grid_rank, grid_coords[2]; // cartesian grid
    MPI_Comm grid_comm;            // communicator with cartesian topology
    MPI_Comm row_comm;
    char* input_file = NULL;       // path to file

    // initialize MPI environment
    MPI_Init(&argc, &argv);
    // get the rank ID of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // print the hello message for each MPI process
    printf("\nHello from rank %d!\n", rank);

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
    double *local_A, *local_B;
    int matrix_size = 0;        // initilize to value beacuse otherwise segmentation fault is triggered

    if (rank == 0) {
        // read input matrices,  pass as reference, only on master process
        read_input_matrices_from_file(input_file, &A, &B, &matrix_size);

        printf("Got matrix size: %d\n", matrix_size);

        // check matrix size
        if (matrix_size % p != 0) {
            printf("[ERROR] The matrix size must be divisible by the root of the number of processes.");

            // quit
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        else {
            printf("Read the two input matrixes as follow:\n");
            printf("Input A:\n");
            print_matrix(A, matrix_size);
            printf("Input B:\n");
            print_matrix(B, matrix_size);
            printf("\n");
        }
    }

    // broadcast matrix_size to all processes, it is needed in fox algorithm
    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate local blocks of A and B on all processes
    local_A = allocate_matrix(TILE_SIZE);
    local_B = allocate_matrix(TILE_SIZE);

    // create cartesian grid, set variables
    int ndims = 2;                  // number of dimensions in grid, always 2D
    int dims[2] = {p, p};           // integer array of size ndims, specifying number of processes in each dimension
    int periods[2] = {1, 1};        // "boolean" array, use periodic boundaries (wrap around) for both dimensions
    int reorder = 1;                // "boolean", let MPI reorder ranks for more efficient process layout
    // initilize a grid for all processes to be in
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, ndims, grid_coords);

    // create a sub-grid for all the processes in the same row of the process grid
    int remain_dims[2] = {0, 1};    // which dimensions to keep, we keep the second dimension or rows
    MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    MPI_Barrier(row_comm);

    // allocate C matrix
    double* local_C = allocate_matrix(TILE_SIZE);
    // allocate full matrices (on rank 0)
    double* C_full = NULL;
    if (rank == 0) {
        // use whole matrix size since it is the full matrix
        C_full = allocate_matrix(matrix_size);
    }

    // distribute the blocks
    distribute_blocks(A, B, local_A, local_B, matrix_size, rank, processes, TILE_SIZE, grid_comm);

    /*
    printf("After scattering, local matrices from rank %d:\n", rank);
    printf("Block A:\n");
    print_matrix(local_A, TILE_SIZE);
    printf("Block B:\n");
    print_matrix(local_B, TILE_SIZE);
    */

    // run fox algorithm
    for (int step = 0; step < p; step++) {
        printf("Fox algorithm running on process %d, step %d:\n", rank, step);

        // calculate root process for this step
        // alt: (rank % p + step) % p
        int root = (grid_coords[0] + step) % p;

        //printf("Root: %d, Grid Coordinates: %d, %d", root, grid_coords[0], grid_coords[1]);

        // broadcast the block A in each row
        MPI_Bcast(local_A, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, root, row_comm);

        /*
        if (grid_coords[1] == root) {
            printf("Broadcasting local_A because of root %d...", root);
        }
        */

        // multiply
        multiply_accumalate(local_A, local_B, local_C, TILE_SIZE);

        /*
        printf("Multiplied the following matrices:\n");
        printf("A\n");
        print_matrix(local_A, TILE_SIZE);
        printf("B\n");
        print_matrix(local_B, TILE_SIZE);

        // debug
        printf("Following matrix was produced (local C):\n");
        print_matrix(local_C, TILE_SIZE);
        */

        printf("\n");

        // shift block B left by one process in its row
        int left, right;
        MPI_Cart_shift(grid_comm, 0, -1, &right, &left);
        MPI_Sendrecv_replace(local_B, TILE_SIZE * TILE_SIZE, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // gather results to form the full matrix C on master process (rank 0)
    gather_results(local_C, C_full, TILE_SIZE, grid_comm);
    
    if (rank == 0) {
        // TODO: compare resulting matrix with answer?
        printf("Final C Matrix:\n");
        print_matrix(C_full, matrix_size);
        // compare
        test_matrix_corectness(C_full, matrix_size, input_file);
    }

    // clean up memory allocations
    free(local_A);
    free(local_B);
    free(local_C);
    // free upp full matrix and input matrix if master process
    if (rank == 0) {
        free(A);
        free(B);
        free(C_full);
    }
 
    // finalize the MPI environment
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}