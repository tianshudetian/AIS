import pypsr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

def heaviside(value,threshold=0.8):
    if value >= threshold:
        return 1
    else:
        return 0

def k_distribute_display(K_set):
    k_distribute = np.zeros(int(K_set.max()))
    for i in np.arange(0, K_set.max(), 1):
        for K in K_set:
            if K == i + 1:
                k_distribute[int(i)] = k_distribute[int(i)] + 1

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = np.arange(1, len(k_distribute) + 1, 1)
    ax.scatter(x, k_distribute, marker='*', c='red')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel('K', fontsize=17, fontname='Times New Roman')
    ax.set_ylabel('The number of nodes', fontsize=17, fontname='Times New Roman')
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')


input_file = r'/home/mty/conflictNUM.csv'
df = pd.read_csv(input_file)
origin_info = list(df.iloc[:, 1])
reconstruct_list = pypsr.reconstruct(origin_info, 12, 6)
k_set=np.zeros(len(reconstruct_list))
threshold = 0.8
warnings.filterwarnings("error")
for index1, vector1 in enumerate(reconstruct_list):
    tem_list = reconstruct_list[index1:, :]
    if len(tem_list) > 1:
        for index2, vector2 in enumerate(tem_list):
            try:
                correlation_coefficient = np.corrcoef(vector1, vector2)
                D = heaviside(correlation_coefficient[0][1], threshold)
                if D == 1:
                    k_set[index1] = k_set[index1] + D
                    k_set[index2+index1+1] = k_set[index2+index1+1] + D
            except:
                pass
K_set = np.array(k_set)
# print(K_set)
# np.save('K_set.npy', K_set)