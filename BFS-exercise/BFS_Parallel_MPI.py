from mpi4py import MPI
import igraph as ig
import random
import math

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        return result, elapsed
    return wrapper

seed = 42
V1 = 1000
V2 = 10000
V3 = 100000

p = 0.01


random.seed(seed) #ksfglkfsjslk


#Generating graphs for the BFS
print("Generating G1")
G1 = ig.Graph.Erdos_Renyi(n=V1, p=p, directed=False, loops=False)
print("Generating G2")
G2 = ig.Graph.Erdos_Renyi(n=V2, p=p, directed=False, loops=False)
print("Generating G3")
G3 = ig.Graph.Erdos_Renyi(n=V3, p=p, directed=False, loops=False)

@timer
def BFS_parallel_MPI(G, start_node):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_nodes = G.vs
    visited = {node: False for node in all_nodes}
    distance = {node: float('inf') for node in all_nodes}

    frontier = []
    if rank == 0:
        visited[start_node] = True
        distance[start_node] = 0
        frontier = [start_node]
 
    frontier_number = 1

    while True:
        # Broadcast the current frontier to all processes
        frontier = comm.bcast(frontier, root=0)
        if not frontier:
            break

        # Calculate slice of frontier for this process
        frontier_len = len(frontier)
        chunk_size = math.ceil(frontier_len / size)
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, frontier_len)
        try:
            local_frontier = frontier[start:end]
        except:
            local_frontier = None

        # Local computation
        local_next_frontier = set()
        for node in local_frontier:
            for neighbor in G.neighbors(node):
                if not visited[neighbor]:
                    local_next_frontier.add(neighbor)

        # Gather all local frontiers at root
        gathered = comm.gather(local_next_frontier, root=0)

        # At root: update visited/distance and prepare next frontier
        if rank == 0:
            next_frontier = set()
            for local in gathered:
                for node in local:
                    if not visited[node]:
                        visited[node] = True
                        distance[node] = frontier_number
                        next_frontier.add(node)
            frontier = list(next_frontier)
        else:
            frontier = None

        frontier_number += 1

    # Broadcast final distances to all processes
    distance = comm.bcast(distance, root=0)
    return distance



"""
Running the code:
mpiexec -n 4 python parallel_bfs.py
# Convert to adjacency list
    G = {node: list(G_nx.neighbors(node)) for node in G_nx.nodes()}
"""