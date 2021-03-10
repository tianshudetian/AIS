from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pypsr
import warnings

def heaviside(value, threshold=0.7):
    if value >= threshold:
        return 1
    else:
        return 0


input_file = r'/home/mty/conflictNUM.csv'
df = pd.read_csv(input_file)
origin_info = list(df.iloc[:, 1])

warnings.filterwarnings("error")
reconstruct_list = pypsr.reconstruct(origin_info, 12, 6)
average_clustering = []
for i in np.arange(0.80, 1., 0.01):
    threshold = round(i, 2)
    print(threshold)
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
    node_set = np.arange(0, len(reconstruct_list)-1, 1)
    G=nx.Graph()
    G.add_nodes_from(node_set)
    G.add_edges_from(node_couple_set)
    average_clustering.append(nx.average_clustering(G))
    # if threshold == 0.85:
    #     fig1 = plt.figure(figsize=(8, 5))
    #     ax = fig1.add_subplot(111)
    #     nx.draw_networkx(G, with_labels=False, node_color='r', node_size=50, alpha=0.7, edge_color='k')
    #     np.save(r'node_couple_set(0.86).npy', np.array(node_couple_set))
    #     plt.axis("off")

np.save(r'clustering.npy', np.array(average_clustering))
fig2 = plt.figure(figsize=(8, 5))
ax = fig2.add_subplot(111)
x = np.arange(0.80, 1., 0.01)
ax.plot(x, average_clustering,
        linestyle='-', linewidth=2, color='#99CCFF',
        marker='^', markeredgecolor='#CCCCFF', markerfacecolor='#FFFFFF', markersize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel(r'$r_c$', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('Clustering coefficient', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()