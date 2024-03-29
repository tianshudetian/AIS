from matplotlib import pyplot as plt
from networkx.algorithms import community
import networkx as nx
import numpy as np
import warnings
import pypsr
import pandas as pd
import json
import community_networkx as cn


def heaviside(value,threshold=0.7):
    if value >= threshold:
        return 1
    else:
        return 0


input_file = r'/home/mty/conflictNUM20181024.csv'
df = pd.read_csv(input_file)

origin_info = list(df.iloc[:, 1])
origin_time = list(df.iloc[:, 2])
new_info = []
new_time = []
for index, row in enumerate(origin_info):
    if index%2 == 0:
        new_info.append(row)
        new_time.append(int(origin_time[index]))
warnings.filterwarnings("error")
reconstruct_list = pypsr.reconstruct(new_info, 12, 6)
threshold = 0.92
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
node_max = max(node_set)
from matrix_point_plot import matrix_point_plot
s = matrix_point_plot(node_couple_set, r'node_plot('+input_file[21:29]+','+str(threshold)+'.png')
s.point_plot()

# with open('network(1001,0.85).json','w') as netjson:
#     json.dump([node_set,node_couple_set], netjson)
# print('degree_centrality: '+str(nx.degree_centrality(G)))
# print('betweenness_centrality: '+str(nx.betweenness_centrality(G)))
# print('closeness_centrality: '+str(nx.closeness_centrality(G)))

# degree_centrality
# degree_centrality = nx.degree_centrality(G)
# degree_centrality_list = []
# for i in np.arange(0, node_max+1, 1):
#     degree_centrality_list.append(degree_centrality[i])
# print('network centrality: '+str(np.mean(degree_centrality_list)))
# fig1 = plt.figure(figsize=(12, 8))
# ax = fig1.add_subplot(111)
# plt.scatter(np.arange(0, node_max+1, 1), degree_centrality_list)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# ax.set_xlabel('Nodes', fontsize=17, fontname='Times New Roman')
# ax.set_ylabel('Degree centrality', fontsize=17, fontname='Times New Roman')
# plt.savefig(r'degree_centrality.png', dpi=600)
# plt.tick_params(labelsize=16)
# plt.grid(linestyle='--')




# clustering = nx.clustering(G)
# node_list = list(clustering)
# print(max(node_list))
# print('degree: '+str(G.degree()))
# print("C: "+str(nx.clustering(G)))
# fig1 = plt.figure(figsize=(12, 8))
# ax = fig1.add_subplot(111)
# nx.draw_networkx(G, with_labels=False, node_color='r', node_size=50, alpha=0.8)
# plt.axis("off")
# plt.savefig(r'complex_networkx.png', dpi=600)
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
# fig4 = plt.figure(figsize=(8, 6))
# ax = fig4.add_subplot(111)
# degree = nx.degree_histogram(G)
# plt.scatter(range(len(degree)), [z/float(sum(degree)) for z in degree])
# # plt.xlabel('')
# plt.savefig(r'degree_distribute(0.85).png', dpi=600)

# communities_generator = community.girvan_newman(G)
# top_level_communities = next(communities_generator)
# next_level_communities = next(communities_generator)
# community_structure = sorted(map(sorted, next_level_communities))
# with open('data01.json', 'w') as fjson:
#     json.dump(partition, fjson)
# with open('node_timestamp.json', 'w') as djson:
#     json.dump(new_time, djson)
plt.show()