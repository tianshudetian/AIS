import json
import matplotlib.pyplot as plt
import numpy as np
with open('data01.json', 'r') as fjson:
     data= json.load(fjson)
with open('node_timestamp.json', 'r') as tjson:
    node_timestamp = json.load(tjson)

def time_match(C):
    time_list = []
    for node in C:
        time_list.append(node_timestamp[node])
    return time_list

C0 = []
C1 = []
C2 = []
C3 = []
C4 = []
C5 = []
C6 = []
max_value = 0
for key in data:
    if data[key] == 0:
        C0.append(int(key))
    elif data[key] == 1:
        C1.append(int(key))
    elif data[key] == 2:
        C2.append(int(key))
    elif data[key] == 3:
        C3.append(int(key))
    elif data[key] == 4:
        C4.append(int(key))
    elif data[key] == 5:
        C5.append(int(key))
    elif data[key] == 6:
        C6.append(int(key))
    else:
        pass
    if int(key) > max_value:
        max_value = int(key)
C_set = [C0, C1, C2, C3, C4, C5, C6]

fig1 = plt.figure(figsize=(12, 8))
ax = fig1.add_subplot(111)
for index, c in enumerate(C_set):
    ax.scatter(time_match(c),np.ones(len(c))+index+1)
# plt.savefig(r'complex_networkx.png', dpi=600)
plt.show()