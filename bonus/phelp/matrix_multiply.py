import numpy as np

# file path
file_path = 'phelp/new_matrix.txt'

# read input dimensions and matrices from the file
with open(file_path, 'r') as file:
    # read the dimension of the matrices
    n = int(file.readline().strip())

    # read the line of numbers for matrix A
    line_A = list(map(int, file.readline().strip().split()))

    # read the line of numbers for matrix B
    line_B = list(map(int, file.readline().strip().split()))

# reshape the lists into matrices
matrix_A = np.array(line_A).reshape(n, n)
matrix_B = np.array(line_B).reshape(n, n)

# perform matrix multiplication
result_matrix = np.dot(matrix_A, matrix_B)

# write the resulting matrix
with open('phelp/result_matrix.txt', 'w') as f:
    for row in result_matrix:
        f.write(' '.join(f'{value:.2f}' for value in row))
        f.write('\n')
