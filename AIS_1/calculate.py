# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:55:33 2020

@author: 24592
"""

import pandas as pd
from math import  cos, sin, asin, sqrt, degrees, atan2, radians
trajFile='D:/论文数据/20181024(2s)Processed.csv'
outputFile1='D:/结果/20181024Data.csv'
traj = pd.read_csv(trajFile,encoding='gbk',engine='python')
r = 6371.393 # 地球平均半径，单位为公里
traj.rename(columns={'lon':'Lon','lat':'Lat'},inplace=True)
traj['lat']=traj.apply(lambda x:radians(x['Lat']),axis=1)
traj['lon']=traj.apply(lambda x:radians(x['Lon']),axis=1)
times = traj.timestamp.unique()
timeInt = times[1] - times[0]
Result = []
for time in times:
    data = traj[traj['timestamp'].isin([time])]
    if data.shape[0] >= 2:
        for row in data.itertuples():
            if data.shape[0] > 1:
                result = pd.DataFrame(columns=['RD'])
                data = data[~data['MMSI'].isin([row.MMSI])]
                result['RD'] = data.apply(lambda x:2*r*asin(sqrt(sin((x['lat']-row.lat)/2)**2+cos(row.lat)*cos(x['lat'])*sin((x['lon']-row.lon)/2)**2)),axis=1)
                result['oRB'] = data.apply(lambda x:(degrees(atan2(sin(x['lon']-row.lon)*cos(x['lat']),cos(row.lat)*sin(x['lat'])-sin(row.lat)*cos(x['lat'])*cos(x['lon']-row.lon)))+360)%360,axis=1)
                result['tRB'] = data.apply(lambda x:(degrees(atan2(sin(row.lon-x['lon'])*cos(row.lat),cos(x['lat'])*sin(row.lat)-sin(x['lat'])*cos(row.lat)*cos(row.lon-x['lon'])))+360)%360,axis=1)
                result['olen'] = row.length
                result['oM'] = row.MMSI
                result['oCOG'] = row.COG
                result['oSOG'] = row.SOG
                tem = data[['MMSI','COG','SOG','length','timestamp']]
                result = pd.concat([result,tem],axis=1)
                Result.append(result)
Result = pd.concat(Result,ignore_index=True)
Result.rename(columns={'length':'tlen','MMSI':'tM','COG':'tCOG','SOG':'tSOG'},inplace=True)
Result.to_csv(outputFile1, sep=',', header=True,index=0)
