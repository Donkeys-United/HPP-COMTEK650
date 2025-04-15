import random
from collections import deque
import igraph as ig
import time
import pickle
from BFS_Parallel_MPI import BFS_parallel_MPI

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

BFS_parallel_MPI(G1, 0)