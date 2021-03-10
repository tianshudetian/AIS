from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv(r'/home/mty/conflictNUM20181024.csv')
origin_info = list(data.iloc[:, 1])
new_info = []
for index, row in enumerate(origin_info):
    if index%2 == 0:
        new_info.append(row)


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(np.arange(0, len(new_info)), new_info)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('Time', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('Conflict number', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()