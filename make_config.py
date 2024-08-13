import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

###############config_variables###############
num_of_nodes = 20
random = True
prob = .3
delay_range=[0, 10]
###############config_variables###############

gap = [0 for i in range(num_of_nodes)]
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#%%
if random:
    node_connection = {}
    connection_delay = {}
    for node in range(num_of_nodes):
        node_connection[node] = []
        connection_delay[node] = []
    for node in range(num_of_nodes):
        for n in range(node + 1, num_of_nodes):
            if np.random.choice([0, 1], p=[1-prob, prob]) == 1:
                delay = np.random.choice(range(delay_range[0], delay_range[1]))
                # delay = max(1,np.random.normal(loc=10.0, scale=5.0, size=None))
                node_connection[node].append(n)
                connection_delay[node].append(delay)
                node_connection[n].append(node)
                connection_delay[n].append(delay)
    print(node_connection)
    print(connection_delay)
    G = nx.Graph(node_connection)
    if len(list(nx.connected_components(G)))==1:
        print("$$$$$$$$$$$SUCCESSFUL$$$$$$$$$$$$")
        with open('config/%s' %num_of_nodes, 'w') as f:
            f.write(json.dumps(node_connection))
            f.write("\n")
            f.write(json.dumps(connection_delay,cls=NpEncoder))
            f.write("\n")
            f.write(json.dumps(gap))

#%%
else:
    node_connection = {}
    connection_delay = {}
    for node in range(num_of_nodes):
        node_connection[node] = []
        connection_delay[node] = []


    # Generate power-law graph
    G = nx.powerlaw_cluster_graph(num_of_nodes, m=2, p=0.1)

    for edge in G.edges:
        # print(edge)
        node_connection[edge[0]].append(edge[1])
        node_connection[edge[1]].append(edge[0])
        delay = np.random.choice(range(delay_range[0], delay_range[1]))
        connection_delay[edge[0]].append(delay)
        connection_delay[edge[1]].append(delay)
    if len(list(nx.connected_components(G)))==1:
        print("$$$$$$$$$$$SUCCESSFUL$$$$$$$$$$$$")
        with open('config/%s_power-law' %num_of_nodes, 'w') as f:
            f.write(json.dumps(node_connection))
            f.write("\n")
            f.write(json.dumps(connection_delay,cls=NpEncoder))
            f.write("\n")
            f.write(json.dumps(gap))

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

