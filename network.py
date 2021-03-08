import numpy as np
import matplotlib.pyplot as plt

K_set = np.load('K_set.npy')
k_distribute = np.zeros(int(K_set.max()))
for i in np.arange(0, K_set.max(), 1):
    for K in K_set:
        if K == i+1:
            k_distribute[int(i)] = k_distribute[int(i)] + 1

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
x = np.arange(1, len(k_distribute)+1, 1)
ax.scatter(x, k_distribute, marker='*', c='red')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('K',fontsize=17,fontname = 'Times New Roman')
ax.set_ylabel('The number of nodes',fontsize=17,fontname = 'Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()
breakpoint1 = 0