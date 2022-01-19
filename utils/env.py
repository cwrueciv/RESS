import networkx as nx
import random
from utils.connectivity_customize import all_pairs_node_connectivity
import pickle
from scipy.stats import lognorm

with open('./data/normalized node weight graph2.pkl', 'rb') as handle:
    weight = pickle.load(handle)


def repairing_time(reliability):
    if reliability <=0.2:
        repair_time = 7
    elif reliability <= 0.8:
        repair_time = 2
    else:
        repair_time = 1

    return repair_time


def get_resilience(G):
    tt = all_pairs_node_connectivity(G)
    graph_resilience = 0
    node_resilience_list = []
    for key in tt:
        node_resilience = sum(tt[key].values()) / (len(tt) - 1)
        node_resilience_list.append(node_resilience)
        graph_resilience += node_resilience * weight[key]
    return graph_resilience, node_resilience_list

def generate_scenario(G, seed):
    random.seed(seed)
    dis = lognorm(0.5, scale=0.35)
    for u, v in G.edges:
        G[u][v]['reliability'] = 1 - dis.cdf(random.uniform(0.0, 1.4))
    return G

def initial_situation(seed=None):
    G = nx.read_gpickle('./data/road network graph2.gpickle')
    G = generate_scenario(G, seed)
    resil, node_resil_list = get_resilience(G)

    return G, resil, node_resil_list

def repairing_process(G, action, inspected_path, pre_resilience=0):

    repair_time = repairing_time(G[action[0]][action[1]]['reliability'])

    G[action[0]][action[1]]['reliability'] = 1
    resil, node_resil_list = get_resilience(G)

    reward = (resil - pre_resilience) / repair_time
    done = False
    # we need to make a deep copy for the current list to determine whether it is finished
    if len(inspected_path) == len(G.edges):
        done = True
    return G, resil, reward, done, node_resil_list
