
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    double x, y, z, pi;
    
    int rank, size, provided; //Initializing variables used by MPI
    int global_count = 0; //Initializing global_count
    int local_count = 0; //Initializing local_count for each variable
    double start_time, stop_time; //Initialize time variables


    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); //Start parallelization

    start_time = MPI_Wtime(); //Start clock as soon as MPI is initialized
    MPI_Comm_size(MPI_COMM_WORLD, &size);  //Get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Get each process rank

    srand(SEED * rank); //seed is multiplied by rank so all threads do not get the same random numbers.

    int iterations_per_process = NUM_ITER / size; //Number of iterations per process

    for (int iter = 0; iter < iterations_per_process; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            local_count++;  //Initialize local_count
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //Sums all local_counts to variable global_count in process with rank 0
    //This becomes a synchronisation point that waits for all processes


    if(rank == 0){
        // Estimate Pi and display the result
        pi = ((double)global_count / (double)NUM_ITER) * 4.0;
        stop_time = MPI_Wtime(); //Stop after calculation is done
        printf("The result is %f\n", pi);
        printf("Elapsed time: %f\n", stop_time - start_time); //print execution time
    }

    MPI_Finalize(); //Stop parallelization
    return 0;
}