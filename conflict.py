import pandas as pd
from multiprocessing import Pool
from hilbert import decode, encode
import matplotlib.pyplot as plt
from matplotlib.path import Path
from hilbertMyself import draw_curve, Knear
import time
from scipy.interpolate import CubicSpline
import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2

def positionConvert(data, Ganularity):
    tem = data.copy()
    tem['newLat'] = tem['lat'].apply(lambda x: ((x-latRange[0])/(latRange[1]-latRange[0]))*pow(2, Ganularity))
    tem['newLon'] = tem['lon'].apply(lambda x: ((x-lonRange[0])/(lonRange[1]-lonRange[0]))*pow(2, Ganularity))
    return tem

def len_dataset(df):
    len_set = df[['MMSI', 'Length']].drop_duplicates(keep='first')
    return len_set

def hilbertEncode(data, Ganularity):
    tem = data.copy()
    tem['latCode'] = tem['newLat'].apply(lambda x: round(x))
    tem['lonCode'] = tem['newLon'].apply(lambda x: round(x))
    pos = tem[['latCode', 'lonCode']].values
    H = encode(pos, Dimension, Ganularity)
    tem['H'] = H
    return tem

def datafilter(data):
    newData = data.loc[(data['SOG'] >= 2.0) & (data['lon'] >= lonRange[0]) &
                       (data['lon'] <= lonRange[-1]) & (data['lat'] >= latRange[0]) &
                       (data['lat'] <= latRange[-1])]
    return newData

def cubic(data, T):
    Timestamps = data['timestamp']
    x = Timestamps - min(Timestamps)
    x_new = np.arange(0, T - min(Timestamps), 1)
    lat = data['lat']
    lon = data['lon']
    cs1 = CubicSpline(x, lat - min(lat))
    cs2 = CubicSpline(x, lon - min(lon))
    lat_new = cs1(x_new) + min(lat)
    lon_new = cs2(x_new) + min(lon)
    return lat_new[-1], lon_new[-1]

def trajectory_display(data1, data2, T, MMSI):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('MMSI: '+str(MMSI),fontsize=14)
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(T, data1[0], marker='*', c='red')
    ax1.scatter(data2['timestamp'], data2['lon'], marker='^', c='blue')
    ax1.set_xlabel('Time',fontsize=14)
    ax1.set_ylabel('Longitude',fontsize=14)
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(T, data1[1], marker='*', c='red')
    ax2.scatter(data2['timestamp'], data2['lat'], marker='^', c='blue')
    ax2.set_xlabel('Time',fontsize=14)
    ax2.set_ylabel('Latitude',fontsize=14)

def cordi(data, timestamp):
    MMSIs = list(set(data['MMSI']))
    cordinate = []
    for index, MMSI in enumerate(MMSIs):
        data = data[data['MMSI'].isin([MMSI])]
        tem1 = data.drop_duplicates(subset='timestamp', keep='first')
        tem2 = tem1.sort_values(by='timestamp')
        temp = tem2.loc[(tem2['timestamp']<=timestamp)]
        if temp.shape[0] > 2:
            latInterpolate, lonInterpolate = cubic(tem2, timestamp)
            cog = list(tem2.COG)[-1]
            cordinate.append([MMSI, latInterpolate, lonInterpolate, cog])
    cordinate = pd.DataFrame(cordinate, columns=['MMSI', 'lat', 'lon', 'cog'])
    return cordinate

def cordinate_display(data):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(data.iloc[:, 2], data.iloc[:, 1], marker='*', c='red')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('latitude')

def encoding_display(data, Ganularity):
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
    brng1 = (degrees(atan2(sin(lon2 - lon1) * cos(lat2), cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1))) + 360) % 360
    brng2 = (degrees(atan2(sin(lon1 - lon2) * cos(lat1), cos(lat2) * sin(lat1) - sin(lat2) * cos(lat1) * cos(lon1 - lon2))) + 360) % 360
    return distance, brng1, brng2

def preConflict(data, K):
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
        dis, brng1, brng2 = dis_brng(lat1, lat2, lon1, lon2)
        Result.append([couple, [dis, brng1, brng2]])
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
            dis, brng1, brng2 = dis_brng(lat1, lat2, lon1, lon2)
            Result.append([[row.MMSI, newRow.MMSI], [dis, brng1, brng2]])
    return Result

def ellipseDomain(a=1.6, b=4):
    x = np.arange(-a, a+0.1, 0.1)
    abs_y = [b*sqrt(1-(round(X, 1)/a)**2) for X in x]
    negative_y = [0-y for y in abs_y]
    point_positive = np.array([[x[i], abs_y[i]] for i in np.arange(len(x))])
    point_negative = np.array([[x[i], negative_y[i]] for i in np.arange(len(x))])
    point = pd.DataFrame(np.vstack((point_negative, point_positive[::-1])))
    Points = point.drop_duplicates()
    return Points

def domain(ship_length, Points):
    boundary = Path(Points.apply(lambda x: x*ship_length))
    return boundary

def conflict_confirmation(point,boundary):
    if boundary.contains_point(point):
        state = 1 # conflict
    else:
        state = 0 # no conflict
    return state

def conflict(couple):
    mmsi1 = couple[0][0]
    len1 = list(len_df[len_df['MMSI'].isin([mmsi1])].Length)[0]
    distance = couple[1][0]*1000 # transform the unit km to the unit m
    brng1 = couple[1][1] # from ship 1 to ship 2
    cog1 = list(cordinate[cordinate['MMSI'].isin([mmsi1])].cog)[0]
    theta = radians(brng1-cog1)
    point = [distance*sin(theta), distance*cos(theta)]
    boundary = domain(len1, basicDomin)
    state = conflict_confirmation(point, boundary)
    if state == 0:
        mmsi2 = couple[0][1]
        len2 = list(len_df[len_df['MMSI'].isin([mmsi2])].Length)[0]
        brng2 = couple[1][2] # from ship 2 to ship 1
        cog2 = list(cordinate[cordinate['MMSI'].isin([mmsi2])].cog)[0]
        theta = radians(cog2-brng2)
        point = [distance * sin(theta), distance * cos(theta)]
        boundary = domain(len2, basicDomin)
        state = conflict_confirmation(point, boundary)
        if state == 1:
            return state
        else:
            pass
    else:
        return state

def conflict_detect(data):
    Indexs = []
    for index, couple in enumerate(data):
        if conflict(couple) == 1:
            Indexs.append(index)
    return len(Indexs)

def indirect(cordinate):
    result = []
    K = 4
    Ganularity = 8
    newTemp = positionConvert(cordinate, Ganularity)
    encodeTemp = hilbertEncode(newTemp, Ganularity)
    preConf = preConflict(encodeTemp, K)
    preConf_num = conflict_detect(preConf)
    if preConf_num >= 1:
        print(preConf_num)
    return preConf_num

# if __name__=="__main__":
df = pd.read_csv(r'/home/mty/data/processed2s/20181001(2s)Processed.csv')
df.rename(columns={'COG':'cog', 'length':'Length'}, inplace=True)
# df = origin_df.iloc[:1000, :]
len_df = len_dataset(df)
Dimension = 2
latRange = [29.55, 30.1]
lonRange = [121.9, 122.45]
basicDomin = ellipseDomain(a=1.6, b=4) # generate the basic domain (L=1)
new_df = datafilter(df)
Timestamps = new_df['timestamp'].unique()
for timestamp in Timestamps:
    df1 = new_df[new_df['timestamp'].isin([timestamp])]

    # df1 = new_df.loc[(new_df['timestamp'] >= timestamp - 60) & (new_df['timestamp'] <= timestamp)]
    # if df1.shape[0] >= 10:
    #     cordinate = cordi(df1, timestamp)
    #     Result = indirect(cordinate)
# b=pd.DataFrame(Result)
# b.to_csv(r'/home/mty/result_test.csv')
# boundary = domain(100, basicDomin)
# cordinate_display(cordinate)
# encoding_display(newTemp)
# plt.show()
# H = hilbertEncode(newTemp)
a=1