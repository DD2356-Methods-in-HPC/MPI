import numpy as np

size = 36

# create two matrices of size
matrix_1 = np.random.randint(0, 100, (size, size))
matrix_2 = np.random.randint(0, 100, (size, size))

# flatten the matrices to 1D
flat_matrix_1 = matrix_1.flatten()
flat_matrix_2 = matrix_2.flatten()

# join the flatten array to a string
str_matrix_1 = ' '.join(map(str, flat_matrix_1))
str_matrix_2 = ' '.join(map(str, flat_matrix_2))

# write the strings to files
with open('phelp/new_matrix.txt', 'w') as f:
    f.write(str(size))
    f.write('\n')
    f.write(str_matrix_1)
    f.write('\n')
    f.write(str_matrix_2)
    f.write('\n')

