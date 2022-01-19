import networkx as nx
import osmnx as ox
import random
import pickle
from scipy.stats import lognorm
import numpy as np

def get_shortest_path(G, start, important_nodes):
    dist = np.inf
    for node in important_nodes:
        dist_temp = nx.dijkstra_path_length(G, start, node)
        if dist_temp < dist:
            dist = dist_temp
    return dist


def get_road_network(case_study):
    if case_study == 'Case I':
        G = ox.graph_from_place('Pomona, California, USA', buffer_dist=-3000, network_type='drive_service',
                                simplify=True)
        G2 = ox.simplification.consolidate_intersections(G, tolerance=0.0001, rebuild_graph=True, reconnect_edges=True)
        node_space = {item: k for k, item in enumerate(G2.nodes)}
        pos = {node_space[k]: (G2.nodes[k]['x'], G2.nodes[k]['y']) for k in G2.nodes}

        G_3 = nx.Graph()
        random.seed(24)
        dis = lognorm(0.5, scale=0.35)
        for u, v, weight in G2.edges:
            G_3.add_edge(node_space[u], node_space[v], weight=G2.get_edge_data(u, v)[0]['length'],
                         reliability=1 - dis.cdf(random.uniform(0.0, 1.4)))

        nx.write_gpickle(G_3, 'data/CaseI_network.gpickle')
        f = open("data/CaseI_node.pkl", "wb")
        pickle.dump(pos, f)
        f.close()

        random.seed(8)
        em_nodes = random.sample(G.nodes, 5)

        node_weight = {}

        for node in G.nodes:
            node_weight[node] = get_shortest_path(G, node, em_nodes)

        for key, values in node_weight.items():
            if values < 1:
                node_weight[key] = 1
            else:
                node_weight[key] = 1/values

        total = sum(node_weight.values(), 0.0)
        normalized_weight = {k: v / total for k, v in node_weight.items()}

        f = open("../data/CaseI_normalized node weight.pkl","wb")
        pickle.dump(normalized_weight,f)
        f.close()

    else:
        pos = 0
        G2 = 0
        print("Under development")
