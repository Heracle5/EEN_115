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

def path_protection(topology_file, traffic_file):
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
    def filter_lists(lists, sub_list):
        # 定义一个空列表用于存储结果
        result = []

        # 遍历给定的列表列表
        for lst in lists:
            # 使用之前定义的函数检查当前列表是否包含子列表的连续元素
            if not has_matching_consecutive_subsequence(lst, sub_list):
                # 如果不包含，将其添加到结果列表中
                result.append(lst)

        # 返回结果列表
        return result

    # 这里使用 has_matching_consecutive_subsequence 函数的定义
    def has_matching_consecutive_subsequence(a, b):
        # 确保a是较长的列表
        if len(a) < len(b):
            a, b = b, a

        # 将列表转换为字符串形式，元素之间用非数字字符分隔，这里使用','
        str_a = ','.join(map(str, a))
        str_b = ','.join(map(str, b))

        # 生成b的所有可能的连续子序列的字符串表示，并检查它们是否在a的字符串表示中
        for i in range(len(b) - 1):
            # 生成连续子序列的字符串表示
            subseq_str = ','.join(map(str, b[i:i + 2]))

            # 检查子序列字符串是否存在于a的字符串表示中
            if subseq_str in str_a:
                return True
        return False
    initial_paths = {}
    inintial_back_up_paths= {}
    for (source, target), paths in ksp.items():
        initial_paths[(source, target)] = random.choice(paths)
        while filter_lists(paths, initial_paths[(source, target)]) == []:
            initial_paths[(source, target)] = random.choice(paths)
        inintial_back_up_paths[(source, target)] = random.choice(filter_lists(paths, initial_paths[(source, target)]))
        #inintial_back_up_paths[(source, target)] = paths[1]

    edge_flows = {(u, v): 0 for u, v in G.edges()}
    for (source, target), path in initial_paths.items():
        flow = traffic_matrix[source - 1][target - 1]
        for i in range(len(path) - 1):
            edge_flows[(path[i], path[i + 1])] += flow

    for (source, target), path in inintial_back_up_paths.items():
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
        current_state_backup = inintial_back_up_paths.copy()


        best_state = current_state
        best_edge_flows = current_edge_flows
        best_max_load = current_max_load
        best_state_backup = current_state_backup

        for i in range(max_iter):
            # 降温
            current_temp *= alpha
            if current_temp < final_temp:
                break

            request = random.choice(list(ksp.keys()))
            new_path = random.choice(ksp[request])
            # if filter_lists(ksp[request], new_path)!=[]:
            #     new_back_up_path = random.choice(filter_lists(ksp[request], new_path))
            while filter_lists(ksp[request], new_path)==[]:
                new_path = random.choice(ksp[request])
            new_back_up_path = random.choice(filter_lists(ksp[request], new_path))
            #     random.choice(ksp[request][:1]+ksp[request][2:]))
            # while has_consecutive_common_elements(new_path, new_back_up_path):
            #     new_back_up_path = random.choice(ksp[request][:1] + ksp[request][2:])
            # if new_path==new_back_up_path:
            #     new_back_up_path = random.choice(ksp[request][:1]+ksp[request][2:])

            new_edge_flows = current_edge_flows.copy()
            flow = traffic_matrix[request[0] - 1][request[1] - 1]
            old_path = current_state[request]
            old_back_up_path = current_state_backup[request]

            for i in range(len(old_path) - 1):
                new_edge_flows[(old_path[i], old_path[i + 1])] -= flow
            for i in range(len(old_back_up_path) - 1):
                new_edge_flows[(old_back_up_path[i], old_back_up_path[i + 1])] -= flow

            for i in range(len(new_path) - 1):
                new_edge_flows[(new_path[i], new_path[i + 1])] += flow
            for i in range(len(new_back_up_path) - 1):
                new_edge_flows[(new_back_up_path[i], new_back_up_path[i + 1])] += flow

            new_max_load = calculate_max_edge_load(new_edge_flows)

            # if new_max_load < current_max_load or random.random() < math.exp((current_max_load - new_max_load) / current_temp):
            if new_max_load < current_max_load:
                current_state = current_state.copy()
                current_state_backup = current_state_backup.copy()
                current_state[request] = new_path
                current_state_backup[request] = new_back_up_path

                current_edge_flows = new_edge_flows
                current_max_load = new_max_load

                if new_max_load < best_max_load:
                    best_state = current_state.copy()
                    best_state_backup = current_state_backup.copy()
                    best_edge_flows = current_edge_flows.copy()
                    best_max_load = new_max_load

        return best_state, best_edge_flows, best_max_load, best_state_backup

    # 退火参数
    initial_temp = 1000
    final_temp = 1
    alpha = 0.999999
    max_iter = 100000

    best_state, best_edge_flows, best_max_load, best_state_backup = simulated_annealing(G, ksp, traffic_matrix, initial_temp, final_temp,
                                                                     alpha, max_iter)
    #print(f"Best max edge load: {best_max_load}")

    def draw_network(G, pos, edge_flows, title):
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)

        edge_labels = {(u, v): f'{d["weight"]}\n{edge_flows.get((u, v), 0)}' for u, v, d in G.edges(data=True)}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
        plt.title(title)
        plt.savefig('objective_data_fig/optimized_network_topology-path-protection'+traffic_file_name+'.png')
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

    for request, path in best_state_backup.items():
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
    spectrum_allocation_backup = {}

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
        spectrum_allocation[request] = {'fsu_index': fsu_index, 'fsu_needed': fsu_needed,'path': path}
    for request_backup, path_backup in best_state_backup.items():
        source, target = request
        flow = traffic_matrix[source - 1][target - 1]
        fsu_index_backup, fsu_needed_backup = allocate_spectrum(request_backup, path_backup, flow, spectrum_usage)
        spectrum_allocation_backup[request_backup] = {'fsu_index': fsu_index_backup, 'fsu_needed': fsu_needed_backup,'path': path_backup}









    def draw_spectrum_allocation(best_state, best_state_backup, spectrum_allocation, spectrum_allocation_backup):
        fig, ax = plt.subplots(figsize=(20, 20))
        color_map = {}
        for i, request in enumerate(spectrum_allocation.keys()):
            color_map[request] = plt.cm.tab20(i % 20)

        edge_to_row = {edge: i for i, edge in enumerate(sorted(spectrum_usage.keys()))}

        for request, alloc in spectrum_allocation.items():
            start_col = alloc['fsu_index']
            num_cols = alloc['fsu_needed']
            path = alloc['path']
            for node in range(len(path) - 1):
                edge = (path[node], path[node + 1])
                row = edge_to_row[edge]
                ax.add_patch(Rectangle((start_col, row), num_cols, 1, color=color_map[request]))
                ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='white', ha='center', va='center')

        for i, request in enumerate(spectrum_allocation_backup.keys()):
            color_map[request] = plt.cm.tab20(i % 20)

        edge_to_row = {edge: i for i, edge in enumerate(sorted(spectrum_usage.keys()))}

        for request, alloc in spectrum_allocation_backup.items():
            start_col = alloc['fsu_index']
            num_cols = alloc['fsu_needed']
            path = alloc['path']
            for node in range(len(path) - 1):
                edge = (path[node], path[node + 1])
                row = edge_to_row[edge]
                ax.add_patch(Rectangle((start_col, row), num_cols, 1, fill=False, color=color_map[request],hatch='//'))
                ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='black', ha='center', va='center')


        ax.set_xlim(0, FSU_CAPACITY)
        ax.set_ylim(0, len(spectrum_usage))
        ax.set_yticks(list(range(1, len(spectrum_usage) + 1)))
        ax.set_yticklabels([f"{edge[0]}:{edge[1]}" for edge in spectrum_usage.keys()])
        ax.set_xlabel('FSU Index')
        ax.set_ylabel('Edge')
        ax.set_title('Spectrum Allocation Grid')
        plt.savefig('objective_data_fig/spectrum_allocation-path-protection' + traffic_file_name + '.png')
        plt.show()

    draw_spectrum_allocation(best_state, best_state_backup, spectrum_allocation, spectrum_allocation_backup)

    with open('objective_data_fig/spectrum_allocation-path-protection'+traffic_file_name+'.pkl', 'wb') as f:
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

    #print(performance_calculation(spectrum_usage,best_state,G,COST_PER_CABLE,FSU_PER_CABLE))
    return performance_calculation(spectrum_usage,best_state,G,COST_PER_CABLE,FSU_PER_CABLE)

if __name__ == '__main__':
    num_fsu_used,highest_load,highest_fsu,average_fsu_used,average_path_length,total_cost=path_protection('G7-topology.txt', 'G7-matrix-5.txt')
    # print("Done")





















