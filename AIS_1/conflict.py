# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 00:45:09 2019

@author: 24592
"""

import pandas as pd
#import numpy as np
from math import  cos, sin, asin, sqrt, degrees, atan2, radians
#from scipy import integrate
'''
()加速度插入在轨迹的第八列,如果traj新增列，需要调整列数
'''
#bounFile1='D:/结果/boundary/max_boun.csv'
#bounFile2='D:/结果/boundary/min_boun.csv'
trajFile='D:/论文数据/20181017(2s)Processed.csv'
outputFile1='D:/结果/20181017Data.csv'
#outputFile2='D:/结果/24confResult.csv'
#‪D:\论文数据\20181001(2s)Processed.csv

#maxBoun = pd.read_csv(bounFile1,encoding='gbk',engine='python')
#minBoun = pd.read_csv(bounFile2,encoding='gbk',engine='python')
traj = pd.read_csv(trajFile,encoding='gbk',engine='python')
traj = traj.loc[traj['length']>=50]
#%%
r = 6371.393 # 地球平均半径，单位为公里
#接近趋势中的时间间隔
#deltaT = 30
traj.rename(columns={'lon':'Lon','lat':'Lat'},inplace=True)
#经纬度转换为弧度值
traj['lat']=traj.apply(lambda x:radians(x['Lat']),axis=1)
traj['lon']=traj.apply(lambda x:radians(x['Lon']),axis=1)
##航速kn转换为m/s
#traj['sog']=traj.apply(lambda x:x['SOG']*(1852/3600),axis=1)
#%%
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
##%%数据分组
##船长50~100，100-200，200-300，300~
#lenBins = [50,100,200,300,10000]
#lenLabels = ['1','2','3','4']
#Result['oLrk'] = pd.cut(Result['olen'],bins=lenBins,labels=lenLabels ,include_lowest=True)
#Result['tLrk'] = pd.cut(Result['tlen'],bins=lenBins,labels=lenLabels ,include_lowest=True)
##根据相对方位分组
#BearingI = 10# 分组间隔的大小
#RBBins = []
#RBLabels = []
#for RBBin in range(0,360+BearingI,BearingI):
#    RBLabel = RBBin+BearingI/2
#    RBBins.append(RBBin)
#    RBLabels.append(RBLabel)
##   删掉list中最后一个值
#del RBLabels[-1]
#Result['oBea'] = pd.cut(Result['oRB'],bins=RBBins,labels=RBLabels ,include_lowest=True)
#Result['tBea'] = pd.cut(Result['tRB'],bins=RBBins,labels=RBLabels ,include_lowest=True)
#Result['RD'] = Result.apply(lambda x:x['RD']*1000,axis=1)
#Result['oRDL'] = Result.apply(lambda x:x['RD']/x['olen'],axis=1)
#Result['tRDL'] = Result.apply(lambda x:x['RD']/x['tlen'],axis=1)
#Result['otID'] = Result.apply(lambda x:x['oM']*10+x['tM']*0.1,axis=1)
#Result['toID'] = Result.apply(lambda x:x['tM']*10+x['oM']*0.1,axis=1)
#Result.dropna(axis=0,how='any',inplace=True)
#Result['oLrk']=Result.oLrk.astype('int64')
#Result['tLrk']=Result.tLrk.astype('int64')
#Result['oBea']=Result.oBea.astype('int64')
#Result['tBea']=Result.tBea.astype('int64')
##%%数据筛选，冲突提取
#df = pd.merge(Result,boun,left_on=['oLrk','oBea'],right_on=['Lrk','Bea'])
#df.rename(columns={'maxCSSD':'omaxCSSD','minCSSD':'ominCSSD'},inplace=True)
#df = pd.merge(df,boun,left_on=['tLrk','tBea'],right_on=['Lrk','Bea'])
#df.rename(columns={'maxCSSD':'tmaxCSSD','minCSSD':'tminCSSD'},inplace=True)
##os筛选
#df1 = df.loc[df.oRDL <= df.omaxCSSD]
#temdata1 = df1.loc[df1.oRDL < df1.ominCSSD]
#lists = temdata1.otID.unique()
#for otID in lists:
#    df1 = df1[~df1['otID'].isin([otID])]
##ts筛选
#df2 = df.loc[df.tRDL <= df.tmaxCSSD]
#temdata2 = df2.loc[df2.tRDL < df2.tminCSSD]
#lists = temdata2.toID.unique()
#for toID in lists:
#    df2 = df2[~df2['toID'].isin([toID])]
#newdf = pd.concat([df1,df2],ignore_index=False)
#newdf.drop_duplicates(inplace=True)
#newdf.reset_index(drop=True,inplace=True)
##%%输出
##部分船舶在感知到他船带来的冲突时，他船并不能感知到这些船舶的冲突
#newdf['id'] = newdf.apply(lambda x:x['oM']+x['tM'],axis=1)    
#conNum=len(newdf.id.unique())
#print('冲突次数：'+str(conNum))
#list1=newdf.oM.unique().tolist()
#list2=newdf.tM.unique().tolist()
#list3=list1+list2
#list3=pd.DataFrame(list3)
#list3.drop_duplicates(keep='first',inplace=True)
#print('参与冲突的船舶数量：'+str(len(list3)))
##%%加速度计算，只计算有冲突的船舶，其实只需要算list2中的船舶就行了
#conf_data = []
#for mmsi in list3[0].tolist():
#    mda = traj[traj['MMSI'].isin([mmsi])]
#    sog = mda.sog.tolist()
#    ACC = []
#    for i in range(mda.shape[0]):
#       if i < mda.shape[0]-1:
#           acc = (sog[i+1]-sog[i])/timeInt
#           ACC.append(acc)
#       else:
#           acc = np.nan
#           ACC.append(acc)
#    mda.insert(loc=8,column='acc',value=ACC,allow_duplicates = True)
#    conf_data.append(mda)
#conf_data = pd.concat(conf_data,ignore_index=False)
#conf_data.fillna(method='ffill',inplace=True)#前向补差
##%%冲突严重度计算
##                   两船的接近程度The degree of approach
#newdf['oDeApp'] = newdf.apply(lambda x:(x['omaxCSSD']-x['oRDL'])/(x['omaxCSSD']-x['ominCSSD']),axis=1)
#newdf['tDeApp'] = newdf.apply(lambda x:(x['tmaxCSSD']-x['tRDL'])/(x['tmaxCSSD']-x['tminCSSD']),axis=1)
#newdf['orb'] = newdf.apply(lambda x:radians(x['oRB']),axis=1)
#newdf['trb'] = newdf.apply(lambda x:radians(x['tRB']),axis=1)
#newdf['oDx'] = newdf.apply(lambda x:x['RD']*sin(x['orb']),axis=1)
#newdf['tDx'] = newdf.apply(lambda x:x['RD']*sin(x['trb']),axis=1)
#newdf['oDy'] = newdf.apply(lambda x:x['RD']*cos(x['orb']),axis=1)
#newdf['tDy'] = newdf.apply(lambda x:x['RD']*cos(x['trb']),axis=1)
##                   两船的接近率The rate of approach
##在deltaT的时间内接近的距离，单位为米
#conf_data['DiApp'] = conf_data.apply(lambda x:x['sog']*deltaT+0.5*x['acc']*(deltaT**2),axis=1)#注意：速度单位选择m/s
##航向转弧度
#conf_data['cog'] = conf_data.apply(lambda x:radians(x['COG']),axis=1)
##根据航向分解距离
#conf_data['Dx'] = conf_data.apply(lambda x:x['DiApp']*sin(x['cog']),axis=1)
#conf_data['Dy'] = conf_data.apply(lambda x:x['DiApp']*cos(x['cog']),axis=1)
#Os = []
#Ts = []
#for row in newdf.itertuples():
#    os = conf_data[(conf_data.MMSI==row.oM)&(conf_data.timestamp==row.time)][['Dx','Dy']]
#    ts = conf_data[(conf_data.MMSI==row.tM)&(conf_data.timestamp==row.time)][['Dx','Dy']]
#    Os.append(os)
#    Ts.append(ts)
#Os = pd.concat(Os,ignore_index=False)
#Ts = pd.concat(Ts,ignore_index=False)
#Os.rename(columns={'Dx':'odx','Dy':'ody'},inplace=True)
#Ts.rename(columns={'Dx':'tdx','Dy':'tdy'},inplace=True)
#Os.reset_index(drop=True,inplace=True)
#Ts.reset_index(drop=True,inplace=True)
#newData = pd.concat([newdf,Os,Ts],axis=1)
##接近率是一定的，不会因为本船与他船的互换而改变
#newData['newDis'] = newData.apply(lambda x:pow((pow((x['oDx']+x['tdx']-x['odx']),2)+pow((x['oDy']+x['tdy']-x['ody']),2)),0.5),axis=1)
##计算接近率，接近率为正，代表两船在相互靠近，绝对值越大，说明接近或远离的趋势越强
#newData['RaApp'] = newData.apply(lambda x:(x['RD']-x['newDis'])/x['RD'],axis=1)
##冲突严重度
##下面两行代码会出现warning,原因是冲突的感知是单方面的
##本船感知到的冲突严重度
#newData['oSeCon'] = newData.apply(lambda x:pow(x['oDeApp'],1+x['RaApp']),axis=1)
##他船感知到的冲突严重度
#newData['tSeCon'] = newData.apply(lambda x:pow(x['tDeApp'],1+x['RaApp']),axis=1)
#data1 = newData[['time','otID','oSeCon','oM','tM']]
#data2 = newData[['time','toID','tSeCon','tM','oM']]
#data1.rename(columns={'otID':'ID','oSeCon':'SeCon'},inplace=True)
#data2.rename(columns={'toID':'ID','tSeCon':'SeCon','tM':'oM','oM':'tM'},inplace=True)
#data3 = pd.concat([data1,data2],ignore_index=False)
#data3.sort_values(by=['time'],ascending=True,inplace=True)
#data3.fillna(0.0,inplace=True)
##%%
#Result1 = []
#for group in data3.groupby('ID'):
#    maxSev = group[1].SeCon.max()#最大冲突严重度
#    result=group[1][group[1].SeCon.isin([maxSev])]
#    if abs(maxSev) > 1.0e-16:
#        x = group[1].time.values
#        Duration = x.max()-x.min()#持续时间
#        if Duration > 1.0e-16:
#            result.insert(0,'Duration',Duration)
#            x=x-x[0]
#            y = group[1].SeCon.values
#            Area = integrate.trapz(y, x)#对序列进行积分,曲线下的面积
#            meanSev = Area/Duration#平均冲突严重度
#            result.insert(0,'Area',Area)
#            result.insert(0,'meanSev',meanSev)
#            Result1.append(result)
#Result1 = pd.concat(Result1,ignore_index=False)
#Result1.rename(columns={'SeCon':'maxSev'},inplace=True)
#Result1.sort_values(by=['time'],ascending=True,inplace=True)
#Result1.reset_index(drop=True,inplace=True)
##%%在结果中匹配经纬度点
#Position = []
#for row in Result1.itertuples():
#    POS = traj[(traj.MMSI==row.oM)&(traj.timestamp==row.time)][['Lon','Lat']]
#    Position.append(POS)
#Position = pd.concat(Position,ignore_index=False)
#Position.reset_index(drop=True,inplace=True)
#Result1 = pd.concat([Result1,Position],axis=1)
##%%输出
#data3.to_csv(outputFile1,sep=',',header=True,index=0)
#Result1.to_csv(outputFile2, sep=',', header=True,index=0)