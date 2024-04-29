import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data(file_path):
    """Reads data from the given text file and returns a dictionary of message sizes and times."""
    data = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # skip empty lines
            if line.strip() == '':
                continue
            # check if the line is a "run" header
            if line.startswith('Run'):
                continue
            else:
                # split the line to get message size and time
                message_size, time = line.split()
                message_size = int(message_size)
                time = float(time)
                
                # add time to the dictionary under the message size
                if message_size not in data:
                    data[message_size] = []
                data[message_size].append(time)
    
    return data

def calculate_average_times(data):
    """Calculates the average times for each message size."""
    average_times = {}
    for message_size, times in data.items():
        average_times[message_size] = sum(times) / len(times)
    
    return average_times

def plot_data(data):
    """Plots the data with message size on the x-axis and average time on the y-axis."""
    # extract message sizes and corresponding average times
    message_sizes = list(data.keys())
    average_times = list(data.values())
    
    # create figure for plot
    plt.figure(figsize=(12, 7))
    plt.grid(True, alpha=0.5)

    # remove top and right spines of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlabel('Message Size (B)')
    plt.ylabel('Average Time (s)')

    # plot the actual data
    plt.plot(message_sizes, average_times, marker='o', alpha=0.6, color="blue")

    # set log scale for the x-axis
    plt.xscale('log')
    
    # customize the x-axis labels to be from 1 to 10 in a log scale
    plt.xticks([10**i for i in range(1, 10)], [f'10^{i}' for i in range(1, 10)])
    
    plt.show()

def calculate_bandwidth_and_latency(message_sizes, average_times):
    """Calculates latency and bandwidth from the data using linear regression (polyfit)."""
    # fit a first-order polynomial (line) to the data using polyfit
    p = np.polyfit(message_sizes, average_times, 1)
    
    # extract slope and y-intercept from the polynomial coefficients
    slope = p[0]  # this is 'r', the reciprocal of the bandwidth
    y_intercept = p[1]  # this is 's', the latency
    
    # calculate latency in milliseconds (convert seconds to milliseconds)
    latency_ms = y_intercept * 1000
    
    # calculate bandwidth in gigabytes per second (GB/s)
    # convert slope to bandwidth in GB/s
    # first, slope is in seconds per byte, so we convert it to bytes per second
    bandwidth_bps = 1 / slope
    # convert from bytes per second to gigabytes per second (1 GB = 10^9 bytes)
    bandwidth_gbps = bandwidth_bps / 1e9
    
    return latency_ms, bandwidth_gbps

def main():
    # change depending on which data to visualize
    # 'ex2_outer_results.txt'
    # 'ex2_intra_results.txt'
    file_path = 'ex2_intra_results.txt'
    
    # step 1: Read the data from the text file
    data = read_data(file_path)
    
    # step 2: Calculate average times for each message size
    average_data = calculate_average_times(data)
    
    # step 3: Plot the data
    plot_data(average_data)

    # Step 4: Polyfit data
    # extract data and send into function
    message_sizes = list(average_data.keys())
    average_times = list(average_data.values())

    latency_ms, bandwidth_gbps = calculate_bandwidth_and_latency(message_sizes, average_times)
    # print results
    print(f"Latency: {latency_ms:.3f} ms")
    print(f"Bandwidth: {bandwidth_gbps:.3f} GB/s")
    
if __name__ == '__main__':
    main()