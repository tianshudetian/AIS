# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:44:11 2019

@author: dn4

时间间隔，速度区间范围按需设置

file number: four
"""

import pandas as pd
import time
from math import radians, cos, sin, asin, sqrt, degrees, atan2
'''
运行代码前需检查下列内容：
(1)输入输出地址
(2)地球半径、时间区间、速度区间、相对方位
'''
inputfile='D:/数据/20181022Processed.csv'
outputfile='D:/数据/20181022CSSD.csv'

df = pd.read_csv(inputfile,encoding='gbk',engine='python')
df = df.drop_duplicates()
r = 6371.393 # 地球平均半径，单位为公里
#%%时间数据分割
#   转换需要的时间区间，以5min(300s)为时间间隔
timeInterval = 1200
TimeStampBins = [] # 区间范围
timeLabels = [] #    区间名称
for new_timestamp in range(int(df.timestamp.min()),int(df.timestamp.max()),timeInterval):
    timeLabel = time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(new_timestamp))
    timeLabels.append(timeLabel)
    TimeStampBins.append(new_timestamp)
TimeStampBins.append(TimeStampBins[-1]+timeInterval)
#   生成分割区间
df['cutLable'] = pd.cut(df['timestamp'], bins=TimeStampBins, labels=timeLabels, include_lowest=True)
#%%速度数据分割
#   速度3~7,7~9,9~11,11~
sogBins = [3,6,9,12,100]
sogLabels = ['1','2','3','4']
df['speedRank'] = pd.cut(df['SOG'],bins=sogBins,labels=sogLabels ,include_lowest=True)
#%%根据相对方位分组
BearingI = 10# 分组间隔的大小
RBBins = []
RBLabels = []
for RBBin in range(0,360+BearingI,BearingI):
    RBLabel = RBBin+BearingI/2
    RBBins.append(RBBin)
    RBLabels.append(RBLabel)
#   删掉list中最后一个值
del RBLabels[-1]
#%%十进制度数转弧度
df['lat']=df.apply(lambda x:radians(x['lat']),axis=1)
df['lon']=df.apply(lambda x:radians(x['lon']),axis=1)
#%%CSSD，时间段-方位-速度等级
#最终结果中应包括相对距离，相对方位，方位等级，速度等级，船舶长度共五个因素
endResult= []#CSSD所在集
#   groupby 时间段
for group1 in df.groupby('cutLable'):
    if group1[1].shape[0]>=2:
        Result = []#每个时间段内所有的相对距离，相对方位，速度等级，船舶长度共四个因素
# =============================================================================
# (1)求相对距离，相对方位，速度等级，船舶长度    
# =============================================================================
        #   根据同一时刻分组
        for group2 in group1[1].groupby('timestamp'):
            # 元组无法更改，所以需要提取出dataframe进行处理
            if group2[1].shape[0]>=2:
                temprorySet1=group2[1]
                temprorySet1=temprorySet1.drop_duplicates(['MMSI'])       
                for row in temprorySet1.itertuples():
                    Set=temprorySet1[temprorySet1['MMSI']!=row.MMSI]
                    #   设置空DataFrame
                    result = pd.DataFrame(columns=['rDistance'])
                    #   相对距离
                    #   distance = 2 * asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2))  * r #单位公里
                    result['rDistance']=Set.apply(lambda x:2*r*asin(sqrt(sin((x['lat']-row.lat)/2)**2+cos(row.lat)*cos(x['lat'])*sin((x['lon']-row.lon)/2)**2)),axis=1)
                    #   相对方位,计算结果为他船相对于row中船舶所在的方位
                    #   brng = (degrees(atan2(sin(radLonB - radLonA) * cos(radLatB), cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(radLonB - radLonA))) + 360) % 360
                    result['rBearing']=Set.apply(lambda x:(degrees(atan2(sin(x['lon']-row.lon)*cos(x['lat']),cos(row.lat)*sin(x['lat'])-sin(row.lat)*cos(x['lat'])*cos(x['lon']-row.lon)))+360)%360,axis=1)
                    #相对方位和相对距离已校验
                    result['length']=row.length#  需保存row中船舶的长度
                    result['speedRank']=row.speedRank#  需保存row中船舶的速度分级
                    result['osMMSI']=row.MMSI
                    tsMMSI=Set.pop('MMSI')# 选出ts的MMSI
                    result.insert(5,'tsMMSI',tsMMSI)#   将ts的MMSI插入结果中
                    if result.empty:
                        continue
                    else:
                        Result.append(result)
        Result = pd.concat(Result,ignore_index=True)
        #   根据方位分组，每一个方位组下每一个速度等级中都应该有一个最小距离
        Result['BRank'] = pd.cut(Result['rBearing'],bins=RBBins,labels=RBLabels ,include_lowest=True)

# =============================================================================
# (2)方位-速度等级,求CSSD
# =============================================================================
    #   groupby方位
    for group3 in Result.groupby('BRank'):
        temprorySet2 = group3[1]
#        if temprorySet2['rDistance'].min() < 1.0e-16:
#            print(temprorySet2)
        #   groupby速度
        for group4 in temprorySet2.groupby('speedRank'):
            temprorySet3 = group4[1]
            #   找出相对距离最小所在的列
            temprorySet4=temprorySet3[temprorySet3['rDistance']==temprorySet3['rDistance'].min()]
            endResult.append(temprorySet4)

endResult = pd.concat(endResult,ignore_index=True)
endResult['rDistance']=endResult.apply(lambda x:x['rDistance']*1000,axis=1)
#   重设ID，代表os与ts的组合，然后剔除各自的MMSI
endResult.to_csv(outputfile, sep=',', header=True,index=0)
#   最终结果为相对距离，单位为米