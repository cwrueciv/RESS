from utils.env import *
import networkx as nx
import matplotlib.pyplot as plt
import time

start_time = time.time()
with open('./data/node position graph2.pkl', 'rb') as handle:
    pos = pickle.load(handle)

seed = 14

G_origini = nx.read_gpickle('./data/road network graph2.gpickle')
G, resil, node_resil_list = initial_situation(seed=seed)

# draw the between centrality map
between_centrality = nx.edge_betweenness_centrality(G_origini, weight='weight')

sort_centrality = sorted(between_centrality.items(), key=lambda x: x[1], reverse=True)

repairing_sequence = [i[0] for i in sort_centrality]


resilience = [resil]
inspected_list = []
for action in repairing_sequence:
    inspected_list.append(action)
    _, resil, reward, done, node_resil_list = repairing_process(G, action=action,
                                                                     pre_resilience=resil,
                                                                     inspected_path=inspected_list)
    print(G)

    print('{} repaired, current resilience{}'.format(action, resil))
    resilience.append(resil)

print('used time', time.time()-start_time)
with open('results/random initial/betweeness centrality_{}.pkl'.format(seed), 'wb') as f:
    pickle.dump(resilience, f)

with open('results/random initial/betweeness centrality actions_{}.pkl'.format(seed), 'wb') as f:
    pickle.dump(inspected_list, f)

## plot centrality and resilience curve
edge, weight = zip(*between_centrality.items())
fig, ax = plt.subplots(figsize=(12, 16))
nx.draw_networkx(G_origini, pos, node_color='black', node_size=10, with_labels=False, edge_cmap=plt.cm.jet,
                 width=6, edge_color=weight,ax=ax)
# plt.savefig('../results/pipe betweeness centrality.png', dpi=1200, bbox_inches='tight')
plt.show()

# draw the color bar for the betweentress
cmap = plt.cm.jet
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(weight), vmax=max(weight)))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Reliability')
# plt.savefig('../results/pipe betweeness colorbar.png', dpi=1200, bbox_inches='tight')
plt.show()

plt.plot(resilience, alpha=0.8)
plt.xlabel('time step')
plt.ylabel('network performance')
plt.tick_params(which='both', direction='in',
                bottom=True, top=True, left=True, right=True)
plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.xlim(0)
# plt.savefig('../results/betweeness centrality repairing strategy.png', dpi=1200, bbox_inches='tight')
plt.show()
