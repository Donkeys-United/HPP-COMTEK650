import random
from collections import deque
import igraph as ig
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
def sequential_BFS(graph, start_node):
    visited = set()
    distance = []
    queue = deque([start_node])
    
    for vertex in graph.vs  :
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

print("BFS G1")
results, time_taken = sequential_BFS(G1, 3)
print("BFS G2")
results2, time_taken2 = sequential_BFS(G2, 3)
print("BFS G3")
results3, time_taken3 = sequential_BFS(G3, 3)

print(f"Sequential BFS for {len(G1.vs)} vertexes was performed in: {time_taken} seconds")
print(f"Sequential BFS for {len(G2.vs)} vertexes was performed in: {time_taken2} seconds")
print(f"Sequential BFS for {len(G3.vs)} vertexes was performed in: {time_taken3} seconds")