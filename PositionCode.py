import pandas as pd
from hilbert import decode, encode
import matplotlib.pyplot as plt
from hilbertMyself import draw_curve
import time
from scipy.interpolate import CubicSpline
import numpy as np

def positionConvert(data):
    tem = data.copy()
    tem['newLat'] = tem['lat'].apply(lambda x: (x-latRange[0])/(latRange[1]-latRange[0])*pow(2, Ganularity))
    tem['newLon'] = tem['lon'].apply(lambda x: (x-lonRange[0])/(lonRange[1]-lonRange[0])*pow(2, Ganularity))
    return tem

def hilbertEncode(data):
    tem = data.copy()
    tem['latCode'] = tem['newLat'].apply(lambda x: round(x))
    tem['lonCode'] = tem['newLon'].apply(lambda x: round(x))
    pos = tem[['latCode', 'lonCode']].values
    H = encode(pos, Dimension, Ganularity)
    tem['H'] = H
    return tem

def datafilter(data):
    timeStamp = int(data.iloc[0:1].timestamp)
    timeArray = time.localtime(timeStamp)
    # hoose the conflict detective time
    time1 = timeStamp + 60 * 60 * 14 - 60 * 60 * timeArray.tm_hour - 60 * timeArray.tm_min - timeArray.tm_sec
    time2 = time1 - 3 * 60
    newData = data.loc[(data['timestamp'] >= time2) & (data['timestamp'] <= time1) & (data['SOG'] >= 2.0) &
                       (data['lon'] >= lonRange[0]) & (data['lon'] <= lonRange[-1]) & (data['lat'] >= latRange[0]) &
                       (data['lat'] <= latRange[-1])]
    time11 = time.localtime(time1)
    otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", time11)
    print('The choosen time: '+str(otherStyleTime1))
    return newData, time1

def cubic(data, T):
    Timestamps = data['timestamp']
    x = Timestamps - min(Timestamps)
    x_new = np.arange(0, T - min(Timestamps), 2)
    lat = data['lat']
    lon = data['lon']
    cs1 = CubicSpline(x, lat - min(lat))
    cs2 = CubicSpline(x, lon - min(lon))
    lat_new = cs1(x_new) + min(lat)
    lon_new = cs2(x_new) + min(lon)
    return lat_new[-1], lon_new[-1]

def cordi(data):
    df = data.copy()
    df1, T = datafilter(df)
    MMSIs = list(set(df1['MMSI']))
    cordinate = []
    for MMSI in MMSIs:
        data = df1[df1['MMSI'].isin([MMSI])]
        tem1 = data.drop_duplicates(subset='timestamp', keep='first')
        tem2 = tem1.sort_values(by='timestamp')
        if tem2.shape[0] > 1:
            latInterpolate, lonInterpolate = cubic(tem2, T)
            cordinate.append([MMSI, latInterpolate, lonInterpolate])
    cordinate = pd.DataFrame(cordinate, columns=['MMSI', 'lat', 'lon'])
    return cordinate

# if __name__=="__main__":
df = pd.read_csv(r'/home/mty/data/dynamic/20181001.csv')
# Timestamps = list(set(df['timestamp']))
# # for timestamp in Timestamps:
# timestamp = Timestamps[10]
# temp = df[df['timestamp'].isin([timestamp])]
Ganularity = 4
Dimension = 2
latRange = [29.55, 30.1]
lonRange = [121.9, 122.45]
cordinate = cordi(df)
newTemp = positionConvert(cordinate)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
draw_curve(ax, Ganularity, Dimension)
# if Ganularity == 3:
ax.set_xticks(range(0, pow(2, Ganularity) + 1, 1))
ax.set_yticks(range(0, pow(2, Ganularity) + 1, 1))
plt.grid()
ax.scatter(newTemp['newLat'], newTemp['newLon'], marker='*', c='red')
plt.show()
# H = hilbertEncode(newTemp)
# a=1
