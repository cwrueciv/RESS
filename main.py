from utils.utils import plot_RoadNetwork
from utils.env import repairing_time, initial_situation
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import chain
from scipy.signal import savgol_filter

def open_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_map(seed=24):
    G, _, _ = initial_situation(seed=seed)

    with open('./data/node position graph2.pkl', 'rb') as handle:
        pos = pickle.load(handle)
    random.seed(8)
    important_nodes = random.sample(G.nodes, 5)
    plot_RoadNetwork(pos, G, important_nodes)

def read_results(location='same initial', seed=24):
    G, _, _ = initial_situation(seed=seed)

    action_space = {i: item for i, item in enumerate(G.edges)}
    # load graph learning results
    graph_learning_result = open_file('./results/{}/RL resilience curve_{}.pkl'.format(location, seed))
    graph_learning_actions = open_file('./results/{}/RL repaire sequence_{}.pkl'.format(location, seed))
    graph = [np.sum(i) for i in graph_learning_result]
    idx = np.argmax(graph)
    graph_resilience = graph_learning_result[idx]
    graph_actions = graph_learning_actions[idx]
    graph_days = [repairing_time(G[action_space[action][0]][action_space[action][1]]['reliability']) for action in
                  graph_actions]
    graph_learning_trajectory = np.repeat(graph_resilience[1:], graph_days)
    # load betweenness results
    betweenness = open_file('./results/{}/betweeness centrality_{}.pkl'.format(location, seed))
    betweenness_actions = open_file('./results/{}/betweeness centrality actions_{}.pkl'.format(location, seed))
    betweenness_days = [repairing_time(G[action[0]][action[1]]['reliability']) for action in betweenness_actions]
    betweenness_trajectory = np.repeat(betweenness[1:], betweenness_days)
    # load GA results
    GA_solution = open_file('./results/{}/GA resilience_seed {}.pkl'.format(location, seed))
    GA_actions = open_file('./results/{}/GA actions_seed {}.pkl'.format(location, seed))

    episode_curve = []
    for episode in GA_solution:
        best_idx = np.argmax([sum(t) for t in episode])
        episode_curve.append(episode[best_idx])

    best_idx = np.argmax([sum(t) for t in episode_curve])
    worst_idx = np.argmin([sum(t) for t in episode_curve])

    best_resilience = episode_curve[best_idx]
    best_action = GA_actions[best_idx]
    best_GA_days = [repairing_time(G[action[0]][action[1]]['reliability']) for action in best_action]
    upper_GA_trajectory = np.repeat(best_resilience[1:], best_GA_days)

    worst_resilience = episode_curve[worst_idx]
    worst_action = GA_actions[worst_idx]
    worst_GA_days = [repairing_time(G[action[0]][action[1]]['reliability']) for action in worst_action]
    lower_GA_trajectory = np.repeat(worst_resilience[1:], worst_GA_days)

    with plt.style.context(['science', 'no-latex', 'grid']):
        plt.figure(figsize=(6, 4))

        plt.fill_between(np.arange(len(lower_GA_trajectory)), upper_GA_trajectory, lower_GA_trajectory, alpha=0.6,
                         label='GA strategy', color='gray')

        print('best GA results', np.sum(upper_GA_trajectory))

        plt.plot(graph_learning_trajectory, label='GCN model', color='red')
        print('GCN-DRL model', np.sum(graph_learning_trajectory))

        plt.plot(betweenness_trajectory, label='BC method', color='black', linestyle='-.')
        print('betweeness centrality', np.sum(betweenness_trajectory))
        plt.legend()

        # plt.ylim(0, 1.2)
        # plt.xlim(0, 137)
        # plt.savefig('./results/{}/comparsion_{}.svg'.format(location, seed))
        plt.show()

def plot_training_process():
    n = 5
    G, _, _ = initial_situation(seed=24)

    action_space = {i: item for i, item in enumerate(G.edges)}

    result = open_file('./results/same initial/RL resilience curve_24.pkl')
    actions = open_file('./results/same initial/RL repaire sequence_24.pkl')
    data = np.array(result)
    days = np.array([[repairing_time(G[action_space[action][0]][action_space[action][1]]['reliability']) for action in action_s] for action_s in actions])
    
    each_resilience = []
    for i in range(len(days)):
        trajectory = np.repeat(data[i, 1:], days[i,:])
        resilience = np.sum(trajectory)
        each_resilience.append(resilience)
    
    #    each_resilience.remove(max(each_resilience))
    #    each_resilience = [each_resilience[i] for i in range(0, len(each_resilience), 2)]
    average_resilience = list(chain.from_iterable([np.mean(each_resilience[i:i + n])] * n for i \
                                                  in range(0, len(each_resilience), n)))
    yhat = savgol_filter(average_resilience, 101, 2)
    # yhat = average_resilience
    with plt.style.context(['science', 'no-latex', 'grid']):
        plt.figure(figsize=(6,4))
        plt.plot(each_resilience, alpha=0.5, color='gray', label='Resilience index of each episode')
        plt.plot(yhat, c='black', linewidth=3, alpha=0.8, label='smoothing result by Savitzky-Golay filter')
        plt.xlabel('Training times')
        plt.ylabel('Resilience Index, RI')

if __name__ == '__main__':
    # plot_map(seed=8)
    # plt.show()
    read_results(location='random initial', seed=14)
    plot_training_process()
    # plt.savefig('./results/same initial/RL training process.tiff', dpi=300, bbox_inches='tight')
    plt.show()
