import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        return result, elapsed
    return wrapper

#Variable
seed = 42
V = 1000
p = 0.3
V1 = 10000
random.seed(seed) #ksfglkfsjslk


#Generating graphs for the BFS
G1 = nx.erdos_renyi_graph(n=V, p=p, seed=seed)
G2 = nx.erdos_renyi_graph(n=V1, p=p, seed=seed)
G3 = nx.erdos_renyi_graph(n=100000, p=p, seed=seed)

# graph1 = nx.draw(G1,with_labels=True, font_weight='bold',node_color='yellow')
# plt.show()
# graph2 = nx.draw(G2,with_labels=True, font_weight='bold',node_color='yellow')
# plt.show()

# tree = nx.bfs_tree(G1, 3)
# print(f"BFS tree: {tree}")
# tree_layout = nx.bfs_layout(tree, 3)
# graph3 = nx.draw(tree, tree_layout, with_labels=True, font_weight='bold')
# #plt.show()

@timer
def sequential_BFS(graph, start_node):
    visited = set()
    distance = []
    queue = deque([start_node])
    
    for vertex in graph.nodes:
        distance.append(float("inf"))
    
    distance[start_node] = 0
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                distance[neighbor] = distance[current] + 1
    
    return visited, distance

results, time_taken = sequential_BFS(G1, 3)
results2, time_taken2 = sequential_BFS(G2, 3)
results3, time_taken3 = sequential_BFS(G3, 3)

# print(f"Visited Verticies: {results[0]}")
# print(f"Distance from source vertex: {results[1]}")
print(f"Sequential BFS for {len(G1.nodes)} vertexes was performed in: {time_taken} seconds")
print(f"Sequential BFS for {len(G2.nodes)} vertexes was performed in: {time_taken2} seconds")
print(f"Sequential BFS for {len(G3.nodes)} vertexes was performed in: {time_taken3} seconds")