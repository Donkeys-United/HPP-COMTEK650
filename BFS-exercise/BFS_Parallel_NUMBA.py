import numpy as np
from numba import njit, prange
from math import ceil
from numba.typed import List
import igraph as ig
import random
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
V1 = 100000
V2 = 10000
V3 = 100000

p = 0.01


random.seed(seed) #ksfglkfsjslk


#Generating graphs for the BFS
print("Generating G1")
G1 = ig.Graph.Erdos_Renyi(n=V1, p=p, directed=False, loops=False)
print("Generating G2")
#G2 = ig.Graph.Erdos_Renyi(n=V2, p=p, directed=False, loops=False)
print("Generating G3")
#G3 = ig.Graph.Erdos_Renyi(n=V3, p=p, directed=False, loops=False)

def build_numba_adjlist(g: ig.Graph):
    n = g.vcount()
    adjlist = List()
    for i in range(n):
        neighbors = g.neighbors(i)
        adjlist.append(np.array(neighbors, dtype=np.int32))
    return adjlist

@timer
@njit(parallel=True)
def bfs_parallel(adjlist, start, num_nodes, num_processors):
    distance = np.full(num_nodes, np.inf)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    frontier = np.full(max_nodes, -1, dtype=np.int32)
    next_frontier = np.full(num_nodes, -1, dtype=np.int32)

    distance[start] = 0
    visited[start] = True
    frontier[0] = start
    frontier_size = 1
    level = 1

    while frontier_size > 0:
        jump_size = ceil(frontier_size / num_processors)
        local_sizes = np.zeros(num_processors, dtype=np.int32)
        local_frontiers = np.full((num_processors, max_nodes), -1, dtype=np.int32)

        for p in prange(num_processors):
            start_idx = p * jump_size
            end_idx = min((p + 1) * jump_size, frontier_size)
            count = 0
            for i in range(start_idx, end_idx):
                current = frontier[i]
                neighbors = adjlist[current]
                for j in range(neighbors.shape[0]):
                    w = neighbors[j]
                    if not visited[w]:
                        already_added = False
                        for k in range(count):
                            if local_frontiers[p, k] == w:
                                already_added = True
                                break
                        if not already_added:
                            local_frontiers[p, count] = w
                            count += 1
            local_sizes[p] = count

        # Combine local frontiers
        next_size = 0
        for p in range(num_processors):
            for i in range(local_sizes[p]):
                w = local_frontiers[p, i]
                if not visited[w]:
                    visited[w] = True
                    distance[w] = level
                    next_frontier[next_size] = w
                    next_size += 1

        for i in range(next_size):
            frontier[i] = next_frontier[i]
        frontier_size = next_size
        level += 1

    return distance

converted_graph = build_numba_adjlist(G1)
#print(converted_graph)

result, time_taken = bfs_parallel(converted_graph, 0, len(converted_graph), 16)
#print(result)
print(time_taken)