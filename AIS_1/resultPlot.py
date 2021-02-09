# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:09:57 2020

@author: dn4
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import webbrowser
from folium.plugins import HeatMap

def data_read(path):
    '''
    文件读取
    '''
    df = []
    for root,dirs,files in os.walk(path):
        for f in files:
            data = pd.read_csv(os.path.join(root,f))
            df.append(data)
    df = pd.concat(df,ignore_index=False)
    return df


df = data_read("D:/0924result")
data = df.loc[(df['Severity']!=0)]
traj = data_read("D:/processed2s")
#%%
point = []
for row in data.itertuples():
    a = traj.loc[(traj['MMSI'].isin([row.MMSI]))&(traj['timestamp'].isin([row.Time1]))]
    a['Severity'] = row.Severity
    point.append(a)
    
HeatMapPoint = pd.concat(point,ignore_index=False)
HeatMapPoint.reset_index(drop=True,inplace=True)
#%%
num = len(HeatMapPoint)
data1 = [[HeatMapPoint['lat'].loc[i],HeatMapPoint['lon'].loc[i],HeatMapPoint['Severity'].loc[i]] for i in range(num)]
latitude = 29.9
longitude = 122.175
m = folium.Map(location=[latitude, longitude],zoom_start=14,tiles='Stamen  Terrain') 
HeatMap(data1).add_to(m)

def lineRef(minlat,maxlat,minlon,maxlon):
    p1 = [minlat,minlon]
    p2 = [maxlat,minlon]
    p3 = [maxlat,maxlon]
    p4 = [minlat,maxlon]
    P = [p1,p2,p3,p4,p1]
    ls = folium.PolyLine(P,color='black',weight=3,opacity=0.7)
    return ls

ls1 = lineRef(minlat = 29.8233333333, maxlat = 29.9460166663, minlon = 122.10, maxlon = 122.25)
ls1.add_to(m)

file_path = r"D:\heatmap.html"
m.save(file_path)     # 保存为html文件
webbrowser.open(file_path)