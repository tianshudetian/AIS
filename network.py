import numpy as np
import matplotlib.pyplot as plt

K_set = np.load('K_set(0.85).npy')
k_distribute = np.zeros(int(K_set.max()))
for i in np.arange(0, K_set.max(), 1):
    for K in K_set:
        if K == i+1:
            k_distribute[int(i)] = k_distribute[int(i)] + 1
Sum = sum(k_distribute)
probability = [k_count/Sum for k_count in k_distribute]

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
x = np.arange(1, len(k_distribute)+1, 1)
ax.scatter(x, probability, marker='*', c='red')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('K', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('Probability', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()