import pandas as pd
from hilbert import decode, encode
import matplotlib.pyplot as plt
from hilbertMyself import draw_curve, Knear
import time
from scipy.interpolate import CubicSpline
import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2

def positionConvert(data):
    tem = data.copy()
    tem['newLat'] = tem['lat'].apply(lambda x: ((x-latRange[0])/(latRange[1]-latRange[0]))*pow(2, Ganularity))
    tem['newLon'] = tem['lon'].apply(lambda x: ((x-lonRange[0])/(lonRange[1]-lonRange[0]))*pow(2, Ganularity))
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
    time2 = time1 - 1 * 60
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
        if tem2.shape[0] > 2:
            latInterpolate, lonInterpolate = cubic(tem2, T)
            cordinate.append([MMSI, latInterpolate, lonInterpolate])
    cordinate = pd.DataFrame(cordinate, columns=['MMSI', 'lat', 'lon'])
    return cordinate

def cordinate_display(data):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(data.iloc[:, 2], data.iloc[:, 1], marker='*', c='red')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('latitude')

def encoding_display(data):
    newTemp = data.copy()
    if Ganularity == 3:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        draw_curve(ax, Ganularity, Dimension)
        ax.set_xticks(range(0, pow(2, Ganularity) + 1, 1))
        ax.set_yticks(range(0, pow(2, Ganularity) + 1, 1))
        plt.grid()
        ax.scatter(newTemp['newLon'], newTemp['newLat'], marker='*', c='red')

def  dis_brng(lat1, lat2, lon1, lon2):
    r = 6371.393 # unit: km
    # The unit of distance is km.
    distance = 2 * asin(sqrt(sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))*r
    # The bearing is the bearing of the other ship's position relative to the ship's position,
    # and the reference direction is due north.
    brng = (degrees(atan2(sin(lon2 - lon1) * cos(lat2),
                          cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1))) + 360) % 360
    return distance, brng

def preConflict(data):
    encodeTemp = data.copy()
    nameList = []
    for index, row in encodeTemp.iterrows():
        x = row.lonCode
        y = row.latCode
        newEncodeTemp = encodeTemp.iloc[(index + 1):, :]
        for newRow in newEncodeTemp.itertuples():
            x_new = newRow.lonCode
            y_new = newRow.latCode
            if abs(x - x_new) <= K and abs(y - y_new) <= K:
                nameList.append([int(row.MMSI), int(newRow.MMSI)])
    Result = []
    for couple in nameList:
        cor1 = cordinate[cordinate['MMSI'].isin([couple[0]])]
        cor2 = cordinate[cordinate['MMSI'].isin([couple[1]])]
        lat1 = radians(cor1.lat)
        lon1 = radians(cor1.lon)
        lat2 = radians(cor2.lat)
        lon2 = radians(cor2.lon)
        dis, brng = dis_brng(lat1, lat2, lon1, lon2)
        Result.append([couple, [dis, brng]])
    return Result

def straConflict(data):
    cordinate = data.copy()
    Result = []
    for index, row in cordinate.iterrows():
        lat1 = radians(row.lat)
        lon1 = radians(row.lon)
        newCordinate = cordinate.iloc[(index + 1):, :]
        for newRow in newCordinate.itertuples():
            lat2 = radians(newRow.lat)
            lon2 = radians(newRow.lon)
            dis, brng = dis_brng(lat1, lat2, lon1, lon2)
            Result.append([[row.MMSI, newRow.MMSI], [dis, brng]])
    return Result

def ellipseDomain(a=3.5, b=8):
    x = np.arange(-a, a+0.1, 0.1)
    abs_y = [b*sqrt(1-(round(X, 1)/a)**2) for X in x]
    negative_y = [0-y for y in abs_y]
    pint_positive = np.array([[x[i], abs_y[i]] for i in np.arange(len(x))])
    point_negative = np.array([[x[i], negative_y[i]] for i in np.arange(len(x))])
    point = np.vstack((point_negative, pint_positive))
    Points = np.array([i for n, i in enumerate(point) if i not in point[:n]])
    return Points

# if __name__=="__main__":
df = pd.read_csv(r'/home/mty/data/dynamic/20181001.csv')
Ganularity = 6
Dimension = 2
K = 2
latRange = [29.55, 30.1]
lonRange = [121.9, 122.45]
cordinate = cordi(df)
# straConf = straConflict(cordinate)

print('count of ship: '+str(len(cordinate)))
newTemp = positionConvert(cordinate)
encodeTemp = hilbertEncode(newTemp)
preConf = preConflict(encodeTemp)

basicDomin = ellipseDomain(a=3.5, b=8)
# cordinate_display(cordinate)
# encoding_display(newTemp)
# plt.show()
# H = hilbertEncode(newTemp)
a=1
