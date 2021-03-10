from matplotlib import pyplot as plt
from networkx.algorithms import community
import networkx as nx
import numpy as np
import warnings
import pypsr
import pandas as pd
import community_networkx as cn


def heaviside(value,threshold=0.7):
    if value >= threshold:
        return 1
    else:
        return 0


input_file = r'/home/mty/conflictNUM.csv'
df = pd.read_csv(input_file)

origin_info = list(df.iloc[:, 1])
new_info = []
for index, row in enumerate(origin_info):
    if index%2 == 0:
        new_info.append(row)
warnings.filterwarnings("error")
reconstruct_list = pypsr.reconstruct(new_info, 12, 6)
threshold = 0.85
node_couple_set = []
for index1, vector1 in enumerate(reconstruct_list):
    tem_list = reconstruct_list[index1:, :]
    if len(tem_list) > 1:
        for index2, vector2 in enumerate(tem_list):
            try:
                correlation_coefficient = np.corrcoef(vector1, vector2)
                D = heaviside(correlation_coefficient[0][1], threshold)
                if D == 1:
                    node_couple_set.append((index1, index2+index1+1))
            except:
                pass
node_set = []
for node_couple in node_couple_set:
    if node_couple[0] not in node_set:
        node_set.append(node_couple[0])
    if node_couple[1] not in node_set:
        node_set.append(node_couple[1])
G = nx.Graph()
G.add_nodes_from(node_set)
G.add_edges_from(node_couple_set)
clustering = nx.clustering(G)
node_list = list(clustering)
print(max(node_list))
fig1 = plt.figure(figsize=(12, 8))
ax = fig1.add_subplot(111)
nx.draw_networkx(G, with_labels=False, node_color='r', node_size=50, alpha=0.8)
plt.savefig(r'complex_networkx.png', dpi=600)
# np.save(r'node_couple_set(0.85).npy', np.array(node_couple_set))
#
# fig2 = plt.figure(figsize=(8, 5))
# ax = fig2.add_subplot(111)
# x = np.arange(0, max(node_list)+1, 1)
# y = []
# for node in x:
#     y.append(clustering[node])
# plt.scatter(x, y, marker='*', c='red')
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# ax.set_xlabel('node', fontsize=17, fontname='Times New Roman')
# ax.set_ylabel('Clustering coefficient', fontsize=17, fontname='Times New Roman')
# plt.savefig(r'clustering_coefficient.png', dpi=600)
# plt.tick_params(labelsize=16)
# plt.grid(linestyle='--')
#
# from community import community_louvain
# fig3 = plt.figure(figsize=(12, 8))
# ax = fig3.add_subplot(111)
# partition = community_louvain.best_partition(G)
# pos = cn.community_layout(G, partition)
# nx.draw(G, pos, node_color=list(partition.values()), with_labels=False, node_size=50, alpha=0.8)
# plt.savefig(r'community_structure.png', dpi=600)
plt.show()