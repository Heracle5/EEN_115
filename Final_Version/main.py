from objective import objective
from benchmark import benchmark
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

def main():
    topology_file = 'G7-topology.txt'
    matrix_file = 'G7-matrix-5.txt'
    num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost=objective(topology_file, matrix_file)
    num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark=benchmark(topology_file, matrix_file)
    print(num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost)
    print(num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark)
    data = [num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost]
    data_benchmark = [num_fsu_used_benchmark,highest_load_benchmark,highest_fsu_benchmark,average_fsu_used_benchmark,average_path_length_benchmark,total_cost_benchmark]

    labels = ['FSU Used', 'Highest Load', 'Highest FSU', 'Average FSU Used', 'Average Path Length', 'Total Cost']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    matrix_file = matrix_file.split('.')[0]

    x = np.arange(1)
    width = 0.35
    ax1.bar(x - width / 2, [highest_load], width, label='Our method')
    ax1.bar(x + width / 2, [highest_load_benchmark], width, label='Benchmark')
    ax1.set_ylabel('Value')
    ax1.set_title('Maximum Link Load Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Maximum Link Load'])
    ax1.legend()

    x = np.arange(len(data) - 1)
    width = 0.35
    ax2.bar(x - width / 2, data[:1] + data[2:], width, label='Our method')
    ax2.bar(x + width / 2, data_benchmark[:1] + data_benchmark[2:], width, label='Benchmark')
    ax2.set_ylabel('Value')
    ax2.set_title('Other Comparisons')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels[:1] + labels[2:])
    for label in ax2.get_xticklabels():
        label.set_fontsize(8)
    ax2.legend()

    plt.tight_layout()
    plt.suptitle('Comparison of Our Method and Benchmark of '+matrix_file, fontsize=10)
    plt.savefig('objective_data_fig/comparison.'+matrix_file+'.png')

    plt.show()





if __name__ == "__main__":
    main()