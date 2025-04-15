import random
import networkx as nx
import pickle

#Variable
seed = 42
V = 100
p = 0.001
V1 = 100000
random.seed(seed)


#Generating graphs for the BFS
#G1 = nx.erdos_renyi_graph(n=V, p=p, seed=seed)
G2 = nx.erdos_renyi_graph(n=V1, p=p, seed=seed)

with open('data.pickle', 'wb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle.dump(G2, f)
