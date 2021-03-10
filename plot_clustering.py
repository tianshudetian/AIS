from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
average_clustering = np.load(r'clustering.npy')
fig2 = plt.figure(figsize=(8, 5))
ax = fig2.add_subplot(111)
x = np.arange(0.80, 1., 0.01)
ax.plot(x, average_clustering,
        linestyle='-', linewidth=2, color='#99CC33',
        marker='s', markeredgecolor='#FF6600', markerfacecolor='#FF6600', markersize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel(r'$\mathrm{r}_\mathrm{c}$', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('Average clustering coefficient', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
x_major_locator = MultipleLocator(0.02)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0.79, 1.00)
plt.show()