from objective import objective
from benchmark import benchmark
from path_protection import path_protection
from shared_protection import shared_protection
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

def main():
    topology_file = 'IT10-topology.txt'
    matrix_file = 'IT10-matrix-1.txt'
    num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost=objective(topology_file, matrix_file)
    num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark=benchmark(topology_file, matrix_file)
    num_fsu_used_path_protection,highest_load_path_protection,highest_fsu_path_protection,average_fsu_used_path_protection,average_path_length_path_protection,total_cost_path_protection=path_protection(topology_file, matrix_file)
    num_fsu_used_shared_protection,highest_load_shared_protection,highest_fsu_shared_protection,average_fsu_used_shared_protection,average_path_length_shared_protection,total_cost_shared_protection=shared_protection(topology_file, matrix_file)
    print(num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost)
    print(num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark)
    print(num_fsu_used_path_protection,highest_load_path_protection,highest_fsu_path_protection,average_fsu_used_path_protection,average_path_length_path_protection,total_cost_path_protection)
    print(num_fsu_used_shared_protection,highest_load_shared_protection,highest_fsu_shared_protection,average_fsu_used_shared_protection,average_path_length_shared_protection,total_cost_shared_protection)
    data = [num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost]
    data_benchmark = [num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark]
    data_path_protection = [num_fsu_used_path_protection,highest_load_path_protection,highest_fsu_path_protection,average_fsu_used_path_protection,average_path_length_path_protection,total_cost_path_protection]
    data_shared_protection = [num_fsu_used_shared_protection,highest_load_shared_protection,highest_fsu_shared_protection,average_fsu_used_shared_protection,average_path_length_shared_protection,total_cost_shared_protection]

    labels = ['FSU Used', 'Highest Load', 'Highest FSU', 'Average FSU Used', 'Average Path Length', 'Total Cost']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    matrix_file = matrix_file.split('.')[0]

    x = np.arange(1)
    width = 0.25
    ax1.bar(x - 1.5*width , [highest_load], width, label='Our method')
    ax1.bar(x-0.5*width, [highest_load_benchmark], width, label='Benchmark')
    ax1.bar(x + 0.5*width , [highest_load_path_protection], width, label='Path Protection')
    ax1.bar(x + 1.5*width, [highest_load_shared_protection], width, label='Shared Protection')
    ax1.set_ylabel('Value')
    ax1.set_title('Maximum Link Load Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Maximum Link Load'])
    ax1.legend()

    x = np.arange(len(data) - 1)
    width = 0.25
    ax2.bar(x - 1.5*width , data[:1] + data[2:], width, label='Our method')
    ax2.bar(x -0.5*width, data_benchmark[:1] + data_benchmark[2:], width, label='Benchmark')
    ax2.bar(x + 0.5*width, data_path_protection[:1] + data_path_protection[2:], width, label='Path Protection')
    ax2.bar(x + 1.5*width, data_shared_protection[:1] + data_shared_protection[2:], width, label='Shared Protection')
    ax2.set_ylabel('Value')
    ax2.set_title('Other Comparisons')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels[:1] + labels[2:])
    for label in ax2.get_xticklabels():
        label.set_fontsize(8)
    ax2.legend()

    plt.tight_layout()
    plt.suptitle('Comparison of Our Method, Benchmark, 1+1 path protection of '+matrix_file, fontsize=8)
    plt.savefig('objective_data_fig/comparison-protection-.'+matrix_file+'.png')

    plt.show()





if __name__ == "__main__":
    main()