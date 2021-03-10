import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import os

def file_list(input_path):
    pathList = os.listdir(input_path)
    file_list = []
    for path in pathList:
        if path[:5] == 'K_set':
            if float(path[7:10]) >= 0.80 and float(path[7:10]) <= 0.99:
                file_list.append(path)
    return file_list

def insert_sort(List):
    for i in range(1,len(List)):
        key = List[i]
        j = i-1
        while j>=0:
            if float(key[7:10])<float(List[j][7:10]):
                List[j+1]=List[j]
            else:
                break
            j-=1
        List[j+1] = key
    return List

a = 0.9
fig = plt.figure(figsize=(8.5, 5))
ax = fig.add_subplot(111)

input_path = r'/home/mty/PycharmProjects/AIS'
filelist = file_list(input_path)
new_filelist = insert_sort(filelist)
for path in new_filelist:
    file_path = os.path.join(input_path,path)
    if os.path.isfile(file_path):
        K_set = np.load(file_path)
        k_distribute = np.zeros(int(K_set.max()))
        for i in np.arange(0, K_set.max(), 1):
            for K in K_set:
                if K == i+1:
                    k_distribute[int(i)] = k_distribute[int(i)] + 1

        cumulative = []
        for k in np.arange(1, len(k_distribute)+1, 1)[::-1]:
            cumulative.append(sum(k_distribute[:k]))

        probability = [cumul/cumulative[0] for cumul in cumulative]
        plt.scatter(np.arange(0, len(probability), 1), powerlaw.logcdf(probability, a), label=path[6:10])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('K', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('log(cumulative distribution)', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
# legend_list = [path[7:10] for path in new_filelist]
ax.legend(loc='best', bbox_to_anchor=(1., 0.55, 0.16, 0.5), mode='expand')
plt.show()
test = 1