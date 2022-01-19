import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from utils.env import initial_situation, repairing_process
import torch.nn.functional as F
from random import sample
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl
import networkx as nx
import pickle

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

###### Initial Road network #####
G_origin = nx.read_gpickle('./data/road network graph2.gpickle')

G = dgl.DGLGraph(G_origin)
G_new = dgl.DGLGraph(G_origin)


###### PARAMS ######
learning_rate = 0.0005
num_episodes = 1500
gamma = 0.5

hidden_layer = 64

egreedy = 0.9
egreedy_final = 0
egreedy_decay = 60000
report_interval = 1

replay_mem_size = 500
batch_size = 10

####################
action_space = {i: item for i, item in enumerate(G_origin.edges)}

# number_of_inputs = (110, 110)
number_of_outputs = len(action_space)


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    return epsilon


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class ConvNet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ConvNet, self).__init__()
        self.conv1 = GraphConv(in_dim, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 64)
        self.regression = nn.Linear(64, number_of_outputs)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # Perform graph convolution and activation function.
        feature = g.ndata['h'].view(-1, 1)
        h = F.relu(self.conv1(g, feature))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        # Calculate graph representation by averaging all the node representations.
        return self.regression(hg)


class QNet_Agent(object):
    def __init__(self):
        self.nn = ConvNet(1, 256).to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)


    def store_model(self):
        torch.save(self.nn.state_dict(), './results/same initial/RL learning with same_500.pt')

    def select_action(self, state, epsilon, inspected_list):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():
                state = state.to(device)
                action_from_nn = self.nn(state)
                action_from_nn[0][inspected_list] = -1
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            random.seed()
            action = sample([i for i in list(action_space.keys()) if i not in inspected_list], 1)[0]

        return action

    def optimize(self):

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        # state = state.to(device)
        state = dgl.batch(state).to(device)
        new_state = dgl.batch(new_state).to(device)
        # new_state = new_state.to(device)
        reward = Tensor([reward]).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)

        new_state_values = self.nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * gamma * max_new_state_values
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1).unsqueeze(0)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


qnet_agent = QNet_Agent()

steps_total = []
memory = ExperienceReplay(replay_mem_size)

solved_after = 0
solved = False

start_time = time.time()
final = []
frames = []
frames_total = 0
final_inspected_list = []
final_resilience_curve = []
for i_episode in range(num_episodes):
    G_nump, resil, node_resil_list = initial_situation(seed=24)
    print('initial resilience', resil)
    reward = resil
    resilience_curve = [resil]
    step = 0
    G.ndata['h'] = torch.Tensor(node_resil_list)

    # for step in range(100):
    inspected_list = []
    while True:
        step += 1
        # action = env.action_space.sample()

        epsilon = calculate_epsilon(frames_total)
        action = qnet_agent.select_action(G, epsilon, inspected_list)
        inspected_list.append(action)
        # new_state, reward, done, info = env.step(action)
        G_nump, resil, reward, done, node_resil_list = repairing_process(G_nump, action=action_space[action],
                                                                    pre_resilience=resil, inspected_path=inspected_list)
        if round(reward, 5) == 0.0:
            reward = -1.0
        else:
            reward = reward
        G_new.ndata['h'] = torch.Tensor(node_resil_list)

        memory.push(G, action, G_new, reward, done)

        qnet_agent.optimize()
        G = G_new
        resilience_curve.append(resil)
        frames_total += 1
        if done or (step > 200):
            if i_episode == num_episodes - 1:
                qnet_agent.store_model()
            final.append(resilience_curve)
            final_inspected_list.append(inspected_list)
            if (i_episode % report_interval == 0):
                print("\n*** Episode %i *** \
                      \n Resilience: %.2f \
                      \n epsilon: %.2f, steps_used: %i"
                      %
                      (i_episode,
                       np.sum(resilience_curve),
                       epsilon,
                       step
                       )
                      )

                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            break


def save_file(filetpath, data):
    with open(filetpath, 'wb') as f:
        pickle.dump(data, f)


save_file('./results/same initial/resilience curve with same_500.pkl', final)
save_file('./results/same initial/repaire sequence with same_500.pkl', final_inspected_list)

# send_email('computation finished\n\nThe graph learning process just finished!')
