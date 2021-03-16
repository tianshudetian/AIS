import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'/home/mty/data/dynamicNofilter/20181001.csv')
minlat = 29.55
maxlat = 30.0335
minlon = 121.85
maxlon = 122.5
tem = df.loc[(df.lat >= minlat) & (df.lat <= maxlat) & (df.lon >= minlon) & (df.lon <= maxlon) & (df.SOG <= 20)]
fig = plt.figure(figsize=(8, 5))
ax = sns.histplot(data=tem, x="SOG", bins=30, stat='density', kde=True)
# ax = sns.distplot(tem['SOG'].values, bins=200, kde=True)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('SOG', fontsize=17, fontname='Times New Roman')
ax.set_ylabel('Probability density', fontsize=17, fontname='Times New Roman')
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()
