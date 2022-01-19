import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random

with open('./data/normalized node weight graph2.pkl', 'rb') as handle:
    weight = pickle.load(handle)

def plot_RoadNetwork(pos, G, em_nodes):
    node_color = ['red' if node in em_nodes else 'black' for node in G.nodes]
    node_size = [3 if node in em_nodes else 1 for node in G.nodes]

    edges, weights = zip(*nx.get_edge_attributes(G, 'reliability').items())

    with plt.style.context(['science', 'no-latex', 'grid']):
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
        edges = nx.draw_networkx_edges(G, pos, edge_color=weights, width=1, alpha=0.8, edge_cmap=plt.cm.coolwarm)
        plt.colorbar(edges)
        plt.axis('off')
        plt.savefig('./results/Initial_plot.tiff', dpi=300)
        plt.show()

def initial_situation():
    G = nx.read_gpickle('./data/road network graph2.gpickle')
    with open('./data/node position graph2.pkl', 'rb') as handle:
        pos = pickle.load(handle)
    random.seed(8)
    important_nodes = random.sample(G.nodes, 5)
    return G,  pos, important_nodes