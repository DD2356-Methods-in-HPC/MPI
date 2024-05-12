# source venv/scripts/activate
# python \data\\visualize.py

import matplotlib.pyplot as plt
import os
import numpy as np

# boolean to tell if we should print polyfit line or not
PLOT_POLYFIT = True
# change to which data file should be read
MATRIX_DATA = 144

# get correct path
current_directory = os.getcwd()

# read data from file
file_name = f"data_{MATRIX_DATA}.txt"
file_path = os.path.join(current_directory, "data", file_name)

# get the values we are intressted in (n-task and runtimes)
tasks = {}
current_task = None
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith("n-tasks") or line.startswith("t-tasks"):
            current_task = line.strip()
            tasks[current_task] = []
        elif line.startswith("Number of processes"):
            runtime = float(line.split("Runtime:")[1].split()[0])
            tasks[current_task].append(runtime)

# calculate average and add to the corresponding task value (number of processes used)
averages = {}
for task, runtimes in tasks.items():
    avg_runtime = sum(runtimes) / len(runtimes)
    averages[task] = avg_runtime

# plot
plt.figure(figsize=(9, 7))

x = []
y = []
# add each value to an x and y axis
for task, avg_runtime in averages.items():
    # offset for process text
    offset = 3
    # get current task  (n-tasks) = VALUE, we need first part hence split
    task_info = task.split("=")
    num_processes = int(task_info[1].split()[0])
    # add to x and y
    x.append(num_processes)
    y.append(avg_runtime)
    if (num_processes == 256):
        offset = -18

    # plot text
    plt.text(num_processes + offset, avg_runtime, f'({num_processes})', color='blue')

plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of Processes')
plt.ylabel('Average Runtime (seconds)')
plt.grid(True)
#plt.show()

# polyfit
def calculate_plolyfit(x, y, degree=1):
    """Calculates function coefficients from the data using linear regression (polyfit)."""
    
    # fit a polynomial of the specified degree to the data using polyfit
    p = np.polyfit(x, y, degree)
    
    return p

polynomial_degree = 2   # if degree is 1, then cut off should be 2
if polynomial_degree == 1:
    cut_off = 2             # 2 because we skip 16 and 36
else:
    cut_off = 0

coefficients = calculate_plolyfit(x[cut_off:], y[cut_off:], polynomial_degree)
# print coefficients
print("coefficients:", coefficients)

# plot the fitted line
model_x = np.array(x[cut_off:])
model_y = np.polyval(coefficients, model_x)
if PLOT_POLYFIT == True:
    plt.plot(model_x, model_y, color='red', label='Fitted Line')

plt.legend(loc='upper right')
plt.show()