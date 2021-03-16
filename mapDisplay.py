# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:12:41 2019

@author: mty
"""
import folium
import pandas as pd
import webbrowser

def map_dis(df,P):
    ##中心点位置
    latitude = 29.9
    longitude = 122.175
    m = folium.Map(location=[latitude, longitude],
                   zoom_start=12,
                   tiles='https://api.mapbox.com/styles/v1/ruianjian/ckkb8d23u0s0u17o5qound3sw/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoicnVpYW5qaWFuIiwiYSI6ImNra2I1dGNxbzExbTIydW5zbG9qeW1qMHoifQ.ASrGcXCsAbiw0u9DSm0IXw',
                   attr='XXX Mapbox Attribution')
    ls = folium.PolyLine(P, color='black')
    ls.add_to(m)
    #'Stamen  Terrain'的效果较好
    '''
    location：tuple或list类型输入，用于控制初始地图中心点的坐标，格式为(纬度，经度)或[纬度，经度]，默认为None
    width：int型或str型，int型时，传入的是地图宽度的像素值；str型时，传入的是地图宽度的百分比，形式为'xx%'。默认为'100%'
    height：控制地图的高度，格式同width
    tiles：str型，用于控制绘图调用的地图样式，默认为'OpenStreetMap'，也有一些其他的内建地图样式，如'Stamen  Terrain'、
        'Stamen Toner'、'Mapbox Bright'、'Mapbox Control Room'等；也可以传入'None'来绘制一个没有风格的朴素地图，或传入一个URL来使用其它的自选osm
    max_zoom：int型，控制地图可以放大程度的上限，默认为18
    attr：str型，当在tiles中使用自选URL内的osm时使用，用于给自选osm命名
    control_scale：bool型，控制是否在地图上添加比例尺，默认为False即不添加
    no_touch：bool型，控制地图是否禁止接受来自设备的触控事件譬如拖拽等，默认为False，即不禁止
    '''
    for group in df.groupby('MMSI'):
        tem = group[1]
        if tem.shape[0] > 2:
            timestamp = group[1].timestamp.values
            Index = []
            for index, time in enumerate(timestamp):
                if index == 0:
                    Index.append(index)
                else:
                    diff = time - timestamp[index-1]
                    if diff > 30:
                        Index.append(index)
            Index.append(index)
            if len(Index) ==2:
                data = group[1]
            else:
                for j,idx in enumerate(Index):
                    if idx > 0:
                        data = group[1][Index[j-1]:idx]
        locations = []
        for row in data.itertuples():
            lacation = [row.lat, row.lon]
            locations.append(lacation)
        if len(locations) >= 1:
            ls = folium.PolyLine(locations, color='gray')
            ls.add_to(m)
    output = "/home/mty/trajectory.html"
    m.save(output)
    webbrowser.open(output)
#%%
df = pd.read_csv(r'/home/mty/data/dynamicNofilter/20181024.csv')
minlat = 29.55
maxlat = 30.0335
minlon = 121.85
maxlon = 122.5
p1 = [minlat, minlon]
p2 = [maxlat, minlon]
p3 = [maxlat, maxlon]
p4 = [minlat, maxlon]
P = [p1, p2, p3, p4, p1]
tem = df.loc[(df.lat>=minlat)&(df.lat<=maxlat)&(df.lon>=minlon)&(df.lon<=maxlon)]
map_dis(tem,P)