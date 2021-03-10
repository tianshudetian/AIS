import numpy as np
import matplotlib.pyplot as plt

K_set = np.load('K_set(24,8,0.80).npy')
# K_set = K_set[:100]
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
x = np.arange(1, len(K_set)+1, 1)
ax.scatter(x, K_set, marker='*', c='red')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('node', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('K', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()