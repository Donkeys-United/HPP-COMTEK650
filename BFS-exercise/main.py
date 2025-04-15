import random
from collections import deque
import igraph as ig
import time
import pickle
#from BFS_Parallel_MPI import BFS_parallel_MPI

#Variable
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

from mpi4py import MPI
import math

def BFS_parallel_MPI(G, start_node):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_nodes = list(G.keys())
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
        local_frontier = frontier[start:end]

        # Local computation
        local_next_frontier = set()
        for node in local_frontier:
            for neighbor in G[node]:
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
"""

BFS_parallel_MPI(G1, 0)