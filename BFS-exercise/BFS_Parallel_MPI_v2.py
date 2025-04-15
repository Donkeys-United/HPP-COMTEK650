from mpi4py import MPI
import igraph as ig
import random
import pickle
from collections import deque, defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuration
NUM_VERTICES = 12
p = 0.3
seed = 42
root = 0

random.seed(seed)

# Rank 0: Generate and partition the graph
if rank == 0:
    print("Generating graph...")
    G = ig.Graph.Erdos_Renyi(n=NUM_VERTICES, p=p, directed=False)
    adj_list = G.get_adjlist()

    # Create full adjacency list
    full_graph = {i: neighbors for i, neighbors in enumerate(adj_list)}

    print("Partitioning graph...")
    # Partition graph among processes
    partitions = [dict() for _ in range(size)]
    for vertex, neighbors in full_graph.items():
        owner = vertex * size // NUM_VERTICES
        partitions[owner][vertex] = neighbors

    # Send partitions
    print("Distributing partitions...")
    for i in range(1, size):
        comm.send(pickle.dumps(partitions[i]), dest=i)
    local_graph = partitions[0]
else:
    # Receive local graph
    data = comm.recv(source=0)
    local_graph = pickle.loads(data)

# Local data structures
local_distances = {v: float('inf') for v in local_graph}
next_frontier = deque()
current_frontier = deque()

# Determine who owns what
def get_owner(vertex):
    return vertex * size // NUM_VERTICES

# Init root
if get_owner(root) == rank:
    local_distances[root] = 0
    current_frontier.append(root)

level = 0
while True:
    send_buf = defaultdict(list)

    # Expand current frontier
    while current_frontier:
        v = current_frontier.popleft()
        for neighbor in local_graph.get(v, []):
            owner = get_owner(neighbor)
            if owner == rank:
                # Local update
                if neighbor not in local_distances or local_distances[neighbor] == float('inf'):
                    local_distances[neighbor] = level + 1
                    next_frontier.append(neighbor)
            else:
                send_buf[owner].append(neighbor)

    # Prepare messages
    send_data = [send_buf[i] for i in range(size)]
    recv_data = comm.alltoall(send_data)

    # Process received nodes
    for neighbor_list in recv_data:
        for node in neighbor_list:
            if node in local_graph and local_distances[node] == float('inf'):
                local_distances[node] = level + 1
                next_frontier.append(node)

    # Check for continuation
    has_work = int(len(next_frontier) > 0)
    global_continue = comm.allreduce(has_work, op=MPI.SUM)
    if global_continue == 0:
        break

    # Move to next level
    current_frontier = next_frontier
    next_frontier = deque()
    level += 1

# Gather distances
all_distances = comm.gather(local_distances, root=0)

if rank == 0:
    # Merge distance dicts
    final_distances = {k: float('inf') for k in range(NUM_VERTICES)}
    for part in all_distances:
        for k, v in part.items():
            final_distances[k] = v
    print(final_distances)
