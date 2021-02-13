import pandas as pd
from hilbert import decode, encode
import matplotlib.pyplot as plt
from hilbertMyself import draw_curve

def positionConvert(data):
    tem = data.copy()
    tem['newLat'] = tem['lat'].apply(lambda x: (x-lat_rang[0])/(lat_rang[1]-lat_rang[0])*pow(2, Ganularity))
    tem['newLon'] = tem['lon'].apply(lambda x: (x-lon_rang[0])/(lon_rang[1]-lon_rang[0])*pow(2, Ganularity))
    return tem

def hilbertEncode(data):
    tem = data.copy()
    tem['latCode'] = tem['newLat'].apply(lambda x: round(x))
    tem['lonCode'] = tem['newLon'].apply(lambda x: round(x))
    pos = tem[['latCode', 'lonCode']].values
    H = encode(pos, Dimension, Ganularity)
    tem['H'] = H
    return tem



# if __name__=="__main__":
df = pd.read_csv(r'/home/mty/data/dynamic/20181001.csv')
Timestamps = list(set(df['timestamp']))
# for timestamp in Timestamps:
timestamp = Timestamps[10]
temp = df[df['timestamp'].isin([timestamp])]
Ganularity = 3
Dimension = 2
lat_rang = [29.55, 30.1]
lon_rang = [121.9, 122.45]
newTemp = positionConvert(temp)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
draw_curve(ax, Ganularity, Dimension)
if Ganularity == 3:
    ax.set_xticks(range(0, pow(2, Ganularity) + 1, 1))
    ax.set_yticks(range(0, pow(2, Ganularity) + 1, 1))
    plt.grid()
    ax.scatter(newTemp['newLat'], newTemp['newLon'], marker='*', c='red')
plt.show()
H = hilbertEncode(newTemp)
a=1
