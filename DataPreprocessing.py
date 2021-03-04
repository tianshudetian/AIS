import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def diff(a:list)->list:
    diff_a = []
    for index, i in enumerate(a):
        if index >= 1:
            diff_a.append([i - a[index - 1]])
    return diff_a

def fig(x,y,x_name,y_name):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.set_xlabel(str(x_name))
    ax.set_ylabel(str(y_name))
df=pd.read_csv(r'/home/mty/data/dynamic/20181001.csv')
MMSIs = list(set(df['MMSI']))
num = 4
mmsi = MMSIs[num]
data = df[df['MMSI'].isin([mmsi])].iloc[50:100, :]
timestamp = data['timestamp'].values
X = timestamp - min(timestamp)
COG = data['lon']
fig(X, COG, 'Time', 'Cog')
diff_X = diff(X)
diff_lon = diff(COG.values)
fig(diff_X, )


# plt.show()
a=1