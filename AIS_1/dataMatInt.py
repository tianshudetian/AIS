# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:44:37 2019

@author: dn4

Purpose: dynamic and static data match

file number: three
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import math
'''
运行代码前需检查下列内容：
(1)输入输出地址
(2)时间戳间隔
'''
#两个输入
dynamicInputFile='D:/处理数据/20170710-14.csv'
#‪D:\数据\20181002.csv
staticInputFile='D:/论文数据/static/舟山201707.csv'
#‪D:\解析数据\201810-12Data.csv
outputfile='D:/论文数据/201707Processed.csv'
dfDy = pd.read_csv(dynamicInputFile,encoding='gbk',engine='python')
# =============================================================================
# 处理17年合并数据时需要使用
dfDy=dfDy[~dfDy.MMSI.isin(['MMSI'])]
dfDy=dfDy.dropna()
dfDy['MMSI']=dfDy.MMSI.astype('int')
dfDy['lat']=dfDy['lat'].astype('float')
dfDy['lon']=dfDy['lon'].astype('float')
dfDy['SOG']=dfDy['SOG'].astype('float')
dfDy['timestamp']=dfDy['timestamp'].astype('float')
dfDy['COG']=dfDy['COG'].astype('float')
# =============================================================================
#统计动态数据中的MMSI数量
lists = dfDy['MMSI'].unique()
len1=len(lists)
#读取静态数据 ‪D:\处理数据\舟山2017年7月静态数据\2_舟山2017年7月静态数据_.csv
dfSt = pd.read_csv(staticInputFile,encoding='gbk',engine='python')
##
time_interval = 10
#%%
df = pd.merge(dfDy,dfSt)
df.sort_values(by=['timestamp'],ascending=True,inplace=True)
list1=df.MMSI.unique()
list2=dfDy.MMSI.unique()
print("There are %d ships here!" %(len(list2)))
print("Matching failure: %d ships " %(len(list2)-len(list1)))
#%% 转换新时间戳，以30s为时间间隔
df['timestamp'] = df.timestamp.astype('int')
newTimeStamp = []
for new_timestamp in range(int(df.timestamp.min()),int(df.timestamp.max()),time_interval):
    newTimeStamp.append(new_timestamp)
newTimeStamp = pd.DataFrame(newTimeStamp)
#%%
df['cosCOG']=df['COG'].map(lambda x: math.cos(math.radians(x)))
df['sinCOG']=df['COG'].map(lambda x: math.sin(math.radians(x)))
#%%
#插值
def int_pol(D,length,MMSI):
    D.reset_index(drop=True,inplace=True)#重设行索引，舍弃原索引，保存新索引
    x = D.timestamp.values
    #   SOG, lon, lat
    y1 = D.SOG.values
    y2 = D.lon.values
    y3 = D.lat.values
    #   cosCOG, sinCOG
    y4 = D.cosCOG.values
    y5 = D.sinCOG.values
    #   生成时间范围内的每一秒时间戳
    xnew = []
    for time_stamp in range(int(min(x)),int(max(x)),1):
        xnew.append(time_stamp)
    xnew = np.array(xnew)
    #   list转换为array
    #SOG,lon,lat,cosCOG,sinCOG插值
    f1 = interpolate.interp1d(x,y1,kind='slinear')
    f2 = interpolate.interp1d(x,y2,kind='slinear')
    f3 = interpolate.interp1d(x,y3,kind='slinear')
    f4 = interpolate.interp1d(x,y4,kind='slinear')
    f5 = interpolate.interp1d(x,y5,kind='slinear')
    y1new = f1(xnew)
    y2new = f2(xnew)
    y3new = f3(xnew)
    y4new = f4(xnew)
    y5new = f5(xnew)
    data = pd.DataFrame({'timestamp':xnew,'SOG':y1new,'lon':y2new,'lat':y3new,'cosCOG':y4new,'sinCOG':y5new})
    data['COG'] = data.apply(lambda x:math.degrees(math.atan2(x['sinCOG'],x['cosCOG'])),axis=1)
    data = data.drop(['cosCOG', 'sinCOG'], axis=1)
    data['length'] = length
    data['MMSI'] = MMSI
    return data
#%%
results = []
for group in df.groupby('MMSI'):#group为元组(MMSI,DataFrame)
    MMSI, Datas = group
    Datas = Datas.drop_duplicates('timestamp')#去除时间戳重复的行，随机保留一行
#    Datas.sort_values(by=['timestamp'],ascending=True,inplace=True)#根据时间戳进行升序排序    
    if len(Datas.timestamp) <2:
        print('不插值的船舶：',MMSI)#MMSI号码为此的船舶数据过少
    else:
        timestamp = Datas.timestamp.values#x为时间戳
        length = Datas.iat[0,7]#提取船长，船长数据位于第7列
        ##轨迹分段
        Index = []#分段节点的索引，包括初始点与终点
        for index,time in enumerate(timestamp):
            if index == 0:
                Index.append(index)#保留初始点的索引
            else:
                diff = timestamp[index]-timestamp[index-1]
                if diff >= 600:
                #如果两个时间戳之间的时间间隔大于10min，默认为轨迹分段
                #单个异常时间戳的存在对最终结果的影响可以忽略
                    Index.append(index)#保留分段点的索引
        Index.append(index)#保留最后一个索引
        if len(Index) == 2:#索引列表中仅有起点索引和终点索引
            D = Datas#直接对数据进行处理
            result = int_pol(D,length,MMSI)
            results.append(result)
        else:#分段处理
            for j,idx in enumerate(Index):#提取分段节点
                if idx > 0:#排除初始点再开始
                    D = Datas[Index[j-1]:idx]
                    if D.shape[0]>3:
                        #不对大小小于3的dataframe进行处理
                        #如果该节点属于异常节点，dataframe可能仅有2行
                        result = int_pol(D,length,MMSI)
                        results.append(result)
Result = pd.concat(results,ignore_index=True)
Result.sort_values(by=['timestamp'],ascending=True,inplace=True)
newTimes = Result['timestamp'].unique()
#将数据按新时间戳进行保存
newData = []
for newTime in newTimes:
    if newTime in newTimeStamp.values:
        Data = Result[Result['timestamp'].isin([newTime])]
        if Data.empty:#判断是否存在空dataframe
            continue
        else:
            newData.append(Data)
newDF = pd.concat(newData,ignore_index=True)
newDF.to_csv(outputfile,sep=',',header=True,index=0)