import numpy as np

# File path
file_path = 'test_debug.txt'

# Read input dimensions and matrices from the file
with open(file_path, 'r') as file:
    # Read the dimension of the matrices
    n = int(file.readline().strip())

    # Read the line of numbers for matrix A
    line_A = list(map(int, file.readline().strip().split()))

    # Read the line of numbers for matrix B
    line_B = list(map(int, file.readline().strip().split()))

# Reshape the lists into matrices
matrix_A = np.array(line_A).reshape(n, n)
matrix_B = np.array(line_B).reshape(n, n)

# Perform matrix multiplication
result_matrix = np.dot(matrix_A, matrix_B)

# Print the resulting matrix
for row in result_matrix:
    print(' '.join(f'{value:.2f}' for value in row))
