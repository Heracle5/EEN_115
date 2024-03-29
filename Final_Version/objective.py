import random
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import Rectangle



def preprocessing_topogy(file_path):
    edges = []
    with open(file_path, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split('\t')
            link_id = int(parts[0])
            node_a = int(parts[3])
            node_b = int(parts[4])
            length = int(parts[5])
            edges.append((node_a, node_b, length))
            edges.append((node_b, node_a, length))  # Adding reverse edge
    return edges

def preprocessing_traffic(file_path):
    traffic_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            row_data = [int(x) for x in parts]
            traffic_matrix.append(row_data)
    return traffic_matrix

def objective(topology_file, traffic_file):
    traffic_file_name = traffic_file.replace('.txt', '')
    edges= preprocessing_topogy(topology_file)
    traffic_matrix= preprocessing_traffic(traffic_file)

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)

    ksp = {}
    for i, j in product(range(7), repeat=2):
        if i != j:
            source, target = i + 1, j + 1
            ksp[(source, target)] = list(nx.shortest_simple_paths(G, source, target, weight='weight'))[:5]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Initial Network Topology")
    plt.show()

    initial_paths = {}
    for (source, target), paths in ksp.items():
        initial_paths[(source, target)] = paths[0]

    edge_flows = {(u, v): 0 for u, v in G.edges()}
    for (source, target), path in initial_paths.items():
        flow = traffic_matrix[source - 1][target - 1]
        for i in range(len(path) - 1):
            edge_flows[(path[i], path[i + 1])] += flow

    # 固定随机种子:2001110011 for IT-Matrix-5:1027
    # 固定随机种子:1989604 for IT-Matrix-4:738
    # 固定随机种子:110101195306153019 for IT-Matrix-3:445 #感谢提供身份证号码的神必人士
    # 固定随机种子:114514 for IT-Matrix-3:238
    # 固定随机种子:19890604 for IT-Matrix-2:84

    random.seed(19890604)

    def calculate_max_edge_load(edge_flows):
        return max(edge_flows.values())

    def simulated_annealing(G, ksp, traffic_matrix, initial_temp, final_temp, alpha, max_iter):
        current_temp = initial_temp
        current_state = initial_paths.copy()
        current_edge_flows = edge_flows.copy()
        current_max_load = calculate_max_edge_load(current_edge_flows)

        best_state = current_state
        best_edge_flows = current_edge_flows
        best_max_load = current_max_load

        for i in range(max_iter):
            # 降温
            current_temp *= alpha
            if current_temp < final_temp:
                break

            request = random.choice(list(ksp.keys()))
            new_path = random.choice(ksp[request][1:])

            new_edge_flows = current_edge_flows.copy()
            flow = traffic_matrix[request[0] - 1][request[1] - 1]
            old_path = current_state[request]

            for i in range(len(old_path) - 1):
                new_edge_flows[(old_path[i], old_path[i + 1])] -= flow

            for i in range(len(new_path) - 1):
                new_edge_flows[(new_path[i], new_path[i + 1])] += flow

            new_max_load = calculate_max_edge_load(new_edge_flows)

            # if new_max_load < current_max_load or random.random() < math.exp((current_max_load - new_max_load) / current_temp):
            if new_max_load < current_max_load:
                current_state = current_state.copy()
                current_state[request] = new_path
                current_edge_flows = new_edge_flows
                current_max_load = new_max_load

                if new_max_load < best_max_load:
                    best_state = current_state.copy()
                    best_edge_flows = current_edge_flows.copy()
                    best_max_load = new_max_load

        return best_state, best_edge_flows, best_max_load

    # 退火参数
    initial_temp = 1000
    final_temp = 1
    alpha = 0.999999
    max_iter = 100000

    best_state, best_edge_flows, best_max_load = simulated_annealing(G, ksp, traffic_matrix, initial_temp, final_temp,
                                                                     alpha, max_iter)
    #print(f"Best max edge load: {best_max_load}")

    def draw_network(G, pos, edge_flows, title):
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)

        edge_labels = {(u, v): f'{d["weight"]}\n{edge_flows.get((u, v), 0)}' for u, v, d in G.edges(data=True)}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
        plt.title(title)
        plt.savefig('objective_data_fig/optimized_network_topology-'+traffic_file_name+'.png')
        plt.show()

    draw_network(G, pos, best_edge_flows, "Optimized Network Topology")

    edge_traffic_details = {}

    for request, path in best_state.items():
        source, target = request
        flow = traffic_matrix[source - 1][target - 1]
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in edge_traffic_details:
                edge_traffic_details[edge] = []
            edge_traffic_details[edge].append({'path': path, 'flow': flow, 'source_target': (source, target)})

    # for edge, details in edge_traffic_details.items():
         #print(f"Edge {edge}:")
    #     for detail in details:
            #print(f"  Path: {detail['path']}, Flow: {detail['flow']}, Request: {detail['source_target']}")

    a = 1 + 1

    FSU_CAPACITY = 320
    CABLE_CAPACITY = 20
    FSU_PER_CABLE = 3
    COST_PER_CABLE = 2

    spectrum_usage = {edge: [0] * FSU_CAPACITY for edge in G.edges()}

    spectrum_allocation = {}

    def allocate_spectrum(request, path, flow, spectrum_usage):
        fsu_needed = int(math.ceil(flow / CABLE_CAPACITY)) * FSU_PER_CABLE
        fsu_availability = [float('inf')] * (FSU_CAPACITY - fsu_needed + 1)

        for start_index in range(FSU_CAPACITY - fsu_needed + 1):
            fsu_sum = 0
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                fsu_sum += sum(spectrum_usage[edge][start_index:start_index + fsu_needed])
            fsu_availability[start_index] = fsu_sum

        min_usage = min(fsu_availability)
        fsu_index = fsu_availability.index(min_usage)

        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            for j in range(fsu_index, fsu_index + fsu_needed):
                spectrum_usage[edge][j] += 1

        return fsu_index, fsu_needed

    for request, path in best_state.items():
        source, target = request
        flow = traffic_matrix[source - 1][target - 1]
        fsu_index, fsu_needed = allocate_spectrum(request, path, flow, spectrum_usage)
        spectrum_allocation[request] = {'fsu_index': fsu_index, 'fsu_needed': fsu_needed}

    def draw_spectrum_allocation(spectrum_usage, spectrum_allocation, edge_traffic_details):
        fig, ax = plt.subplots(figsize=(20, 20))

        # 颜色映射...
        color_map = {}
        for i, request in enumerate(spectrum_allocation.keys()):
            color_map[request] = plt.cm.tab20(i % 20)

        edge_to_row = {edge: i for i, edge in enumerate(sorted(spectrum_usage.keys()))}

        for edge, details in edge_traffic_details.items():
            for detail in details:
                request = detail['source_target']
                alloc = spectrum_allocation.get(request)
                if alloc:
                    start_col = alloc['fsu_index']
                    num_cols = alloc['fsu_needed']
                    if num_cols == 0:
                        continue
                    row = edge_to_row[edge]
                    ax.add_patch(Rectangle((start_col, row), num_cols, 1, color=color_map[request]))
                    # ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='white', ha='center', va='center'), 字体修改得小一些
                    ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='white', ha='center', va='center')

        ax.set_xlim(0, FSU_CAPACITY)
        ax.set_ylim(0, len(spectrum_usage))
        ax.set_yticks(list(range(1, len(spectrum_usage)+1)))
        ax.set_yticklabels([f"{edge[0]}:{edge[1]}" for edge in spectrum_usage.keys()])
        ax.set_xlabel('FSU Index')
        ax.set_ylabel('Edge')
        ax.set_title('Spectrum Allocation Grid')
        plt.savefig('objective_data_fig/spectrum_allocation-'+traffic_file_name+'.png')
        plt.show()

    draw_spectrum_allocation(spectrum_usage, spectrum_allocation, edge_traffic_details)

    with open('objective_data_fig/spectrum_allocation-'+traffic_file_name+'.pkl', 'wb') as f:
        pickle.dump(spectrum_allocation, f)
        pickle.dump(edge_traffic_details, f)
        pickle.dump(spectrum_usage, f)


    def performance_calculation(spectrum_usage,best_state,G,COST_PER_CABLE,FSU_PER_CABLE):
        num_fsu_used = 0
        highest_load=0
        highest_fsu=0
        total_length=0
        total_cost=0
        for request in spectrum_usage:
            num_fsu_used+=sum(spectrum_usage[request])
            if sum(spectrum_usage[request])>highest_load:
                highest_load=sum(spectrum_usage[request])
            try:
                if (len(spectrum_usage[request]) - spectrum_usage[request][::-1].index(1))>highest_fsu:
                    highest_fsu=(len(spectrum_usage[request]) - spectrum_usage[request][::-1].index(1))
            except:
                pass
        average_fsu_used=math.ceil(num_fsu_used/len(spectrum_usage))
        for request in best_state:
            for i in range(len(best_state[request])-1):
                total_length+=G[best_state[request][i]][best_state[request][i+1]]['weight']

        average_path_length=math.ceil(total_length/len(best_state))
        total_cost=(num_fsu_used/FSU_PER_CABLE)*COST_PER_CABLE
        return num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost

    # print(performance_calculation(spectrum_usage,best_state,G,COST_PER_CABLE,FSU_PER_CABLE))
    return performance_calculation(spectrum_usage,best_state,G,COST_PER_CABLE,FSU_PER_CABLE)

if __name__ == '__main__':
    num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost=objective('G7-topology.txt', 'G7-matrix-5.txt')
    # print("Done")





















