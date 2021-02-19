import pandas as pd
import numpy as np
import time
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from math import cos, sin, radians
from Curvature import curvature

def datafilter(data):
    timeStamp = int(data.iloc[0:1].timestamp)
    timeArray = time.localtime(timeStamp)
    # hoose the conflict detective time
    time1 = timeStamp + 60 * 60 * 14 - 60 * 60 * timeArray.tm_hour - 60 * timeArray.tm_min - timeArray.tm_sec
    time2 = time1 - 3 * 60
    newData = data.loc[(data['timestamp'] >= time2) & (data['timestamp'] <= time1) & (data['SOG'] >= 2.0 )]
    time11 = time.localtime(time1)
    otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", time11)
    print('time1: '+str(otherStyleTime1))
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

filepath = r'/home/mty/data/dynamic/20181001.csv'
df = pd.read_csv(filepath)
df1, T = datafilter(df)
MMSIs = list(set(df1['MMSI']))
# for MMSI in MMSIs:
MMSI = MMSIs[0]
data = df1[df1['MMSI'].isin([MMSI])]
tem = data.drop_duplicates(subset='timestamp', keep='first')
tem.sort_values(by='timestamp', inplace=True)
# latInterpolate, lonInterpolate = cubic(tem, T)
# lon = tem['lon']
# lat = tem['lat']
# X = (lon - min(lon))/(max(lon)-min(lon))+0.1
# Y = (lat - min(lat))/(max(lat)-min(lat))+0.1
# x = X.iloc[-1]
# y = Y.iloc[-1]
# V = tem['SOG'].iloc[-1] #velocity
# Vchange = V/(60*3600)
# C = tem['COG'].iloc[-1] #course
# t1 = T - tem['timestamp'].iloc[-1]
# LAT = lat.iloc[-1] #latitude
# dot_x = Vchange*sin(radians(C))
# dot_y = Vchange*cos(radians(C))/cos(radians(LAT))
# # calculate an
# x1 = list(X.iloc[-4:-1])
# y1 = list(Y.iloc[-4:-1])
# kappa = curvature(x1, y1)
# r = -2/kappa
# an = Vchange**2./r
# # lculate at
# diff_velocity = (V - tem['SOG'].iloc[-2])/(60*3600)
# if diff_velocity == 0:
#     at = 0.0
# else:
#     diff_time = tem['timestamp'].iloc[-1] - tem['timestamp'].iloc[-2]
#     at = diff_velocity/diff_time

adadada=1321