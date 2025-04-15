from mpi4py import MPI
import igraph as ig
import random
import pickle
from collections import deque, defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === Graph parameters ===
seed = 42
NUM_VERTICES = 100
p = 0.1
root = 0

random.seed(seed)

# === Rank 0 generates graph and partitions ===
if rank == 0:
    print("Generating Graph...")
    G = ig.Graph.Erdos_Renyi(n=NUM_VERTICES, p=p, directed=False, loops=False)
    adj_list = G.get_adjlist()

    full_graph = {i: neighbors for i, neighbors in enumerate(adj_list)}

    # Partition graph by assigning vertices to processes
    partitions = [dict() for _ in range(size)]
    for vertex, neighbors in full_graph.items():
        owner = vertex * size // NUM_VERTICES  # consistent ownership rule
        partitions[owner][vertex] = neighbors

    for i in range(size):
        if i == 0:
            local_graph = partitions[0]
        else:
            comm.send(pickle.dumps(partitions[i]), dest=i)

else:
    local_graph = pickle.loads(comm.recv(source=0))

# === Setup local state ===
visited = set()
local_distances = {}
local_frontier = deque()

# Determine root owner correctly
root_owner = root * size // NUM_VERTICES
if rank == root_owner:
    visited.add(root)
    local_distances[root] = 0
    local_frontier.append(root)

# === BFS loop ===
while True:
    send_buf = defaultdict(list)

    while local_frontier:
        v = local_frontier.popleft()

        # Defensive check: we should only process vertices we own
        if v not in local_graph:
            continue

        for neighbor in local_graph[v]:
            owner = neighbor * size // NUM_VERTICES
            new_dist = local_distances[v] + 1

            if owner == rank:
                if neighbor not in visited:
                    visited.add(neighbor)
                    local_distances[neighbor] = new_dist
                    local_frontier.append(neighbor)
            else:
                send_buf[owner].append((neighbor, new_dist))

    # Prepare data for all-to-all communication
    send_data = [send_buf[i] for i in range(size)]
    recv_data = comm.alltoall(send_data)

    any_new = False
    for items in recv_data:
        for node, dist in items:
            if node not in visited:
                visited.add(node)
                local_distances[node] = dist
                local_frontier.append(node)
                any_new = True

    # Synchronize: are we done globally?
    global_continue = comm.allreduce(int(any_new), op=MPI.SUM)
    if global_continue == 0:
        break

# === Gather and print results ===
all_local_distances = comm.gather(local_distances, root=0)

if rank == 0:
    final_distances = {}
    for proc_distances in all_local_distances:
        final_distances.update(proc_distances)
    print(final_distances)
