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
(2)地球半径、时间区间、长度区间、相对方位
'''
inputfile='D:/论文数据/usedCSSD/20181018Processed.csv'
outputfile='D:/论文数据/20181018CSSD.csv'

df = pd.read_csv(inputfile,encoding='gbk',engine='python')
df = df.drop_duplicates()
r = 6371.393 #地球平均半径，单位为公里
#%%时间数据分割
#转换需要的时间区间，以5min(300s)为时间间隔
timeInterval = 1200
TimeStampBins = [] # 区间范围
timeLabels = [] #    区间名称
for new_timestamp in range(int(df.timestamp.min()),int(df.timestamp.max()),timeInterval):
    timeLabel = time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(new_timestamp))
    timeLabels.append(timeLabel)
    TimeStampBins.append(new_timestamp)
TimeStampBins.append(TimeStampBins[-1]+timeInterval)
#生成分割区间
df['cutLable'] = pd.cut(df['timestamp'], bins=TimeStampBins, labels=timeLabels, include_lowest=True)
#%%船长分割
#船长50~100，100-200，200-300，300~
lenBins = [50,100,200,300,10000]
lenLabels = ['1','2','3','4']
df['lenRank'] = pd.cut(df['length'],bins=lenBins,labels=lenLabels ,include_lowest=True)
#%%根据相对方位分组
BearingI = 10# 分组间隔的大小
RBBins = []
RBLabels = []
for RBBin in range(0,360+BearingI,BearingI):
    RBLabel = RBBin+BearingI/2
    RBBins.append(RBBin)
    RBLabels.append(RBLabel)
#删掉list中最后一个值
del RBLabels[-1]
#%%十进制度数转弧度
df['lat']=df.apply(lambda x:radians(x['lat']),axis=1)
df['lon']=df.apply(lambda x:radians(x['lon']),axis=1)
#%%CSSD
'''
(1)将数据按照时间分段进行分组
(2)再根据某一时刻对分组数据进行分组
(3)计算每一艘船与其他船舶的相对距离,相对方位(该相对方位为两个经纬度间的相对方位)
    记录本船的SOG,length,lenRank,osMMSI,tsMMSI
(4)将相对方位转换为他船相对于本船的相对方位
'''

Result1= []#初始结果，为他船与本船的相对距离……
for group1 in df.groupby('cutLable'):#时间段分组
    if group1[1].shape[0]>=2:
        for group2 in group1[1].groupby('timestamp'):#根据同一时刻分组
            if group2[1].shape[0] >= 2:
                temprorySet1=group2[1]
                temprorySet1=temprorySet1.drop_duplicates(['MMSI'])       
                for row in temprorySet1.itertuples():
                    Set=temprorySet1[temprorySet1['MMSI']!=row.MMSI]
                    result = pd.DataFrame(columns=['rDistance'])#设置空DataFrame
                    #相对距离
                    #distance = 2 * asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2))  * r #单位公里
                    result['rDistance']=Set.apply(lambda x:2*r*asin(sqrt(sin((x['lat']-row.lat)/2)**2+cos(row.lat)*cos(x['lat'])*sin((x['lon']-row.lon)/2)**2)),axis=1)
                    #相对方位,计算结果为他船相对于row中船舶所在的方位
                    #brng=(degrees(atan2(sin(radLonB-radLonA)*cos(radLatB),cos(radLatA)*sin(radLatB)-sin(radLatA)*cos(radLatB)*cos(radLonB-radLonA)))+360)%360
                    result['azimuth']=Set.apply(lambda x:(degrees(atan2(sin(x['lon']-row.lon)*cos(x['lat']),cos(row.lat)*sin(x['lat'])-sin(row.lat)*cos(x['lat'])*cos(x['lon']-row.lon)))+360)%360,axis=1)
                    result['COG']=row.COG
                    #相对方位和相对距离已校验
                    result['length']=row.length#需保存row中船舶的长度
                    result['lenRank']=row.lenRank#需保存row中船舶的长度分组
                    result['osMMSI']=row.MMSI
                    tsMMSI=Set.pop('MMSI')#选出ts的MMSI
                    result.insert(6,'tsMMSI',tsMMSI)#将ts的MMSI插入结果中
                    result['cutLable']=row.cutLable
                    Result1.append(result)
Result2 = pd.concat(Result1,ignore_index=True)
            
            

#将两个经纬度点间的相对方位转换为他船相对于本船的相对方位
Result2['rB'] = Result2.apply(lambda x:x['azimuth']-x['COG'],axis=1)
#
tem_Re1 = Result2[Result2['rB']<0]
tem_Re1_1=tem_Re1.copy()
tem_Re1_1['rBearing'] = tem_Re1.apply(lambda x:x['rB']+360,axis=1)             
#
tem_Re2 = Result2[Result2['rB']>=0]
tem_Re2_1=tem_Re2.copy()
tem_Re2_1['rBearing'] = tem_Re2['rB']
new_Result = pd.concat([tem_Re1_1,tem_Re2_1])
#根据方位分组，每一个方位组下每一个速度等级中都应该有一个最小距离
new_Result['BRank'] = pd.cut(new_Result['rBearing'],bins=RBBins,labels=RBLabels ,include_lowest=True)
#%%
'''
(1)时间段分组
(2)方位分组
(3)长度分组
'''
endResult = []
for group3 in new_Result.groupby('cutLable'):
    for group4 in group3[1].groupby('BRank'):
        for group5 in group4[1].groupby('lenRank'):
            temprorySet3 = group5[1]
            #找出相对距离最小所在的列
            temprorySet4=temprorySet3[temprorySet3['rDistance']==temprorySet3['rDistance'].min()]
            endResult.append(temprorySet4)
endResult = pd.concat(endResult,ignore_index=True)
endResult['rDistance']=endResult.apply(lambda x:x['rDistance']*1000,axis=1)
#重设ID，代表os与ts的组合，然后剔除各自的MMSI
endResult.to_csv(outputfile, sep=',', header=True,index=0)
#最终结果为相对距离，单位为米