import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

from matplotlib.patches import Rectangle

edges = [
    (1, 2, 114), (1, 3, 120), (2, 3, 157), (2, 4, 306), (3, 4, 298),
    (3, 5, 258), (3, 6, 316), (4, 5, 174), (5, 6, 353), (5, 7, 275),
    (6, 7, 224), (2, 1, 114), (3, 1, 120), (3, 2, 157), (4, 2, 306),
    (4, 3, 298), (5, 3, 258), (6, 3, 316), (5, 4, 174), (6, 5, 353),
    (7, 5, 275), (7, 6, 224)
]

G = nx.DiGraph()
G.add_weighted_edges_from(edges)

traffic_matrix = [
    [0, 164, 149, 223, 183, 190, 205],
    [198, 0, 136, 200, 187, 179, 154],
    [202, 195, 0, 259, 232, 297, 132],
    [220, 190, 229, 0, 182, 160, 185],
    [174, 170, 236, 203, 0, 238, 165],
    [199, 193, 212, 242, 166, 0, 204],
    [273, 207, 109, 230, 224, 181, 0]
]

directory_for_variables = {}
for i, j in product(range(7), repeat=2):
    if i != j:
        source, target = i + 1, j + 1
        directory_for_variables[(source, target)] = list(nx.shortest_simple_paths(G, source, target, weight='weight'))[:5]
        
# compute the position of every node in a graph
pos = nx.spring_layout(G)
print(pos)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Initial Network Topology")
plt.show()

initial_paths = {}
for (source, target), paths in directory_for_variables.items():
    initial_paths[(source, target)] = paths[0]

edge_flows = {(u, v): 0 for u, v in G.edges()}
print(edge_flows)
for (source, target), path in initial_paths.items():
    flow = traffic_matrix[source - 1][target - 1]
    for i in range(len(path) - 1):
        edge_flows[(path[i], path[i + 1])] += flow
        print(edge_flows)

import random
import math

random.seed(1000)


def calculate_max_edge_load(edge_flows):
    return max(edge_flows.values())


def simulated_annealing(G, ksp, traffic_matrix):
 
    current_state = initial_paths.copy()
    current_edge_flows = edge_flows.copy()
    current_max_load = calculate_max_edge_load(current_edge_flows)

    best_state = current_state
    best_edge_flows = current_edge_flows
    best_max_load = current_max_load


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

        #if new_max_load < current_max_load or random.random() < math.exp((current_max_load - new_max_load) / current_temp):
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





best_state, best_edge_flows, best_max_load = simulated_annealing(G, directory_for_variables, traffic_matrix)
print(f"Best max edge load: {best_max_load}")


def draw_network(G, pos, edge_flows, title):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)

    edge_labels = {(u, v): f'{d["weight"]}\n{edge_flows.get((u, v), 0)}' for u, v, d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    plt.title(title)
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

for edge, details in edge_traffic_details.items():
    print(f"Edge {edge}:")
    for detail in details:
        print(f"  Path: {detail['path']}, Flow: {detail['flow']}, Request: {detail['source_target']}")

a=1+1



FSU_CAPACITY = 320
CABLE_CAPACITY = 20
FSU_PER_CABLE = 3


spectrum_usage = {edge: [0] * FSU_CAPACITY for edge in G.edges()}

#Initialize an Empty Dictionary:
spectrum_allocation = {}

def allocate_spectrum(request, path, flow, spectrum_usage):
    #calculate needed fsu
    fsu_needed = int(math.ceil(flow / CABLE_CAPACITY)) * FSU_PER_CABLE
    #calculate fsu availability
    fsu_availability = [float('inf')] * (FSU_CAPACITY - fsu_needed + 1)
  # subtracting the needed FSUs from the total FSU capacity.
    for start_index in range(FSU_CAPACITY - fsu_needed + 1):
        fsu_sum = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            fsu_sum += sum(spectrum_usage[edge][start_index:start_index + fsu_needed])
        fsu_availability[start_index] = fsu_sum
 #determined to ensure that there are enough consecutive FSUs for the required amount
    min_usage = min(fsu_availability)
    fsu_index = fsu_availability.index(min_usage)
    #Retrieves the index of the minimum value, representing the optimal start index for allocating FSUs.
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
    fig, ax = plt.subplots(figsize=(50, 50))

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
                #ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='white', ha='center', va='center'), 字体修改得小一些
                ax.text(start_col + num_cols / 2, row + 0.5, str(request), color='white', ha='center', va='center')

    ax.set_xlim(0, FSU_CAPACITY)
    ax.set_ylim(0, len(spectrum_usage))
    ax.set_xlabel('FSU Index')
    ax.set_ylabel('Edge')
    ax.set_title('Spectrum Allocation Grid')
    plt.savefig('spectrum_allocation_GE_1.png')
    plt.show()

draw_spectrum_allocation(spectrum_usage, spectrum_allocation,edge_traffic_details)




