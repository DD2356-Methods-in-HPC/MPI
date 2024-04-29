#!/bin/bash -l

#SBATCH -J task4
#SBATCH -t 1:00:00
#SBATCH -A edu24.DD2356
#SBATCH -p main
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH -e error_file.e

srun -n 8 ./a.out > output

srun -n 16 ./a.out >> output

srun -n 32 ./a.out >> output

srun -n 64 ./a.out >> output

srun -n 128 ./a.out >> output

srun -n 256 ./a.out >> output