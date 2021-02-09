# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:49:59 2020

@author: 24592
"""

import pandas as pd
import scipy.interpolate as spi
from matplotlib.path import Path
import numpy as np
import math

def ra2de(data):
    '''
    (1)弧度转度数，使用math.degrees转化的度数有负值，不符合插值需求
    (2)因数据构成原型，所以最后一个方位，虽然是正数，却需要+360°
    '''
    data['RB'] = data.apply(lambda x:math.degrees(x['Rad']),axis=1)
    RB = []
    for i in Lrk:
        tem = data[data['Lrk']==i]
        rbs = []
        for row in tem.itertuples():
            if row.RB < 0:
                rb = row.RB + 360
                rbs.append(rb)
            else:
                rb = row.RB
                rbs.append(rb)
        rbs[-1]=rbs[-1] + 360
        RB.extend(rbs)
    data['RB'] = RB
    return data

def new_boun(data):
    '''
    使用三次样条插值，对边界数据进行插值，使每一度范围内都存在一个边界数据
    '''
    new_boun = []
    for i in Lrk:
        tem = data[data['Lrk']==i]
        dx = tem.dx.values
        dy = tem.dy.values
        rb = tem.RB.values
        new_rb = np.array(range(0,360))
        f1 = spi.splrep(rb,dx,k=3) #样本点导入，生成参数
        f2 = spi.splrep(rb,dy,k=3) #样本点导入，生成参数
        new_dx = spi.splev(new_rb,f1)
        new_dy = spi.splev(new_rb,f2)
        tem1 = pd.DataFrame({'RB':new_rb,'dx':new_dx,'dy':new_dy})
        tem1['Lrk'] = i
        new_boun.append(tem1)
    new_boun = pd.concat(new_boun,ignore_index=True)
    return new_boun

def dxy2dis(data):
    '''
    dx,dx转换为距离
    '''
    data['Dis'] = data.apply(lambda x:math.sqrt(math.pow(x['dx'],2)+math.\
        pow(x['dy'],2)),axis=1)
    return data

def boun(data1,data2):
    '''
    生成最后所需的边界数据
    '''
    tem1 = ra2de(data1)
    new_maxB = new_boun(tem1)
    new_maxB = dxy2dis(new_maxB)
    new_maxB.rename(columns={'Dis':'maxDis','RB':'RB(r)'},inplace=True)
    tem2 = ra2de(data2)
    new_minB = new_boun(tem2)
    new_minB = dxy2dis(new_minB)
    new_minB.rename(columns={'Dis':'minDis','RB':'RB(r)'},inplace=True)
    return new_maxB,new_minB



def path_boun(data):
    '''
    输入最大边界，生成边界范围
    '''
    for i in Lrk:
        data1 = data[data['Lrk']==i]
        data1 = data1.reset_index(drop=True)
        points = []
        for row in data1.itertuples():
            point = (row.dx,row.dy)
            points.append(point)
            if i == 1:
                p1 = Path(points)
            elif i == 2:
                p2 = Path(points)
            elif i == 3:
                p3 = Path(points)
            else:
                p4 = Path(points)
    return p1,p2,p3,p4

def azi2azi(df):
    '''
    将相对方位转换为相对方位
    '''
    df['orB'] = df.apply(lambda x:x['oRB']-x['oCOG'],axis=1)
    df['trB'] = df.apply(lambda x:x['tRB']-x['tCOG'],axis=1)
    tem_Re1 = df[df['orB']<0]
    tem_Re1_1=tem_Re1.copy()
    tem_Re1_1['orBearing'] = tem_Re1.apply(lambda x:x['orB']+360,axis=1)             
    #
    tem_Re2 = df[df['orB']>=0]
    tem_Re2_1=tem_Re1.copy()
    tem_Re2_1['orBearing'] = tem_Re2['orB']
    new_df = pd.concat([tem_Re1_1,tem_Re2_1])
    ###
    tem_Re3 = new_df[new_df['trB']<0]
    tem_Re3_1=tem_Re3.copy()
    tem_Re3_1['trBearing'] = tem_Re3.apply(lambda x:x['trB']+360,axis=1)             
    #
    tem_Re4 = new_df[new_df['trB']>=0]
    tem_Re4_1=tem_Re4.copy()
    tem_Re4_1['trBearing'] = tem_Re4['trB']
    new_df1 = pd.concat([tem_Re3_1,tem_Re4_1])
    new_df1['oRad'] = new_df1.apply(lambda x:math.radians(x['orBearing']),axis=1)
    new_df1['tRad'] = new_df1.apply(lambda x:math.radians(x['trBearing']),axis=1)
    new_df1['oDis'] = new_df1.apply(lambda x:1000*x['RD']/x['olen'],axis=1)
    new_df1['tDis'] = new_df1.apply(lambda x:1000*x['RD']/x['tlen'],axis=1)
    new_df1['odx'] = new_df1.apply(lambda x:x['oDis']*math.sin(x['oRad']),axis=1)
    new_df1['ody'] = new_df1.apply(lambda x:x['oDis']*math.cos(x['oRad']),axis=1)
    new_df1['tdx'] = new_df1.apply(lambda x:x['tDis']*math.sin(x['tRad']),axis=1)
    new_df1['tdy'] = new_df1.apply(lambda x:x['tDis']*math.cos(x['tRad']),axis=1)
    #new_df1['ID'] = new_df1.apply(lambda x:x['oM']+x['tM'],axis=1)
    return new_df1

def len_cut(df):
    '''
    长度切割
    '''
    lenBins = [50,100,200,300,10000]
    lenLabels = ['1','2','3','4']
    df['oLrk'] = pd.cut(df['olen'],bins=lenBins,labels=lenLabels ,include_lowest=True)
    df['tLrk'] = pd.cut(df['tlen'],bins=lenBins,labels=lenLabels ,include_lowest=True)
    df.dropna(axis=0,how='any',inplace=True)
    return df

def state_ana(Lrk,point):
    '''
    状态分析
    1代表存在冲突，0则反之
    '''
    if (int(Lrk) == 1 and p1.contains_points([point])[0]==True) or\
    (int(Lrk) == 2 and p2.contains_points([point])[0]==True) or\
    (int(Lrk) == 3 and p3.contains_points([point])[0]==True) or\
    (int(Lrk) == 4 and p4.contains_points([point])[0]==True):
        state = 1
    else:
        state = 0
    return state

def add_state(df):
    '''
    为df添加状态
    '''
    oStates = []
    tStates = []
    for row in df.itertuples():
        oPoint = (row.odx,row.ody)
        tPoint = (row.tdx,row.tdy)
        oState = state_ana(row.oLrk,oPoint)
        tState = state_ana(row.tLrk,tPoint)
        oStates.append(oState)
        tStates.append(tState)
    df['oState'] =oStates
    df['tState'] =tStates
    return df
 
def deal_df(df):
    '''
    冲突数据提取与边界匹配
    '''
    df= df.loc[(df['oState']==1) | (df['tState']==1)]
    tem1 = df[df['oState'].isin([1])]
    result1 = tem1[['oDis','oLrk','oRad','orBearing','oM','tM','timestamp']]
    result1_1 = result1.copy()
    result1_1.rename(columns={'oDis':'Dis','oLrk':'Lrk','oRad':'Rad',\
                              'orBearing':'RB'},inplace=True)
    
    tem2 = df[df['tState'].isin([1])]
    result2 = tem2[['tDis','tLrk','tRad','trBearing','tM','oM','timestamp']]
    result2_1 = result2.copy()
    result2_1.rename(columns={'tDis':'Dis','tLrk':'Lrk','tRad':'Rad','trBearing':\
                              'RB','tM':'oM','oM':'tM'},inplace=True)
    result = pd.concat([result1_1,result2_1],ignore_index=False)
    result['ID'] = result.apply(lambda x:x['oM']*10+x['tM']*0.1,axis=1)
    result['RB(r)'] = result['RB'].apply(lambda x:round(x))#航向四舍五入
    result['Lrk'] = result.Lrk.astype('int')
    tem3=new_maxB[['RB(r)','Lrk','maxDis']]
    tem4=new_minB[['RB(r)','Lrk','minDis']]
    result=pd.merge(result,tem3)
    result=pd.merge(result,tem4)
    result['deCon']=result.apply(lambda x:(x['maxDis']-x['Dis'])/(x['maxDis']-\
      x['minDis']),axis=1)
    result.loc[result['deCon']<0,'deCon'] = 0#将计算小于0的值替换为0
    result.sort_values(by='timestamp',ascending=True,inplace=True)
    return result

def con_MMSI(df):
    '''
    输出参与冲突的船舶的MMSI号，及数量
    '''
    list1=df.oM.unique()
    list2=df.tM.unique()
    list3=np.concatenate((list1,list2),axis=0)
    list3 = pd.DataFrame(list3)
    list3.drop_duplicates(inplace=True)
    return list3,len(list3)

def needed_traj(df,List):
    '''
    根据参与冲突船舶的MMSI号，从轨迹数据中筛选出所需要的轨迹数据
    '''
    new_traj = []
    for row in List.itertuples():
        mmsi = row._1
        tem = df[df['MMSI'].isin([mmsi])]
        new_traj.append(tem)
    new_traj = pd.concat(new_traj,ignore_index=True)
    new_traj['sog']=new_traj.apply(lambda x:x['SOG']*(1852/3600),axis=1)
    return new_traj

def acc_cal(data):
    '''
    计算加速度，时间间隔为2s
    '''
    sog = data.sog.tolist()
    diff_sog = np.diff(sog)
    acc = diff_sog/2#2s间隔
    acc = np.insert(acc,-1,acc[-1])#ndarray添加值的方法
    data1 = data.copy()
    data1['acc'] = acc
    return data1

def acc_merge(df):
    '''
    为轨迹数据融合加速度数据，需进行轨迹分段处理
    '''
    result = []
    for group in df.groupby('MMSI'):
        data = group[1]
        time = data.timestamp.tolist()
        Index = []
        for index,i in enumerate(time):
            if index == 0:
                Index.append(index)
            elif index < len(time)-1:
                diff = time[index+1] - time[index]
                if diff > 2:
                    Index.append(index)
            else:
                Index.append(index+1)
                
        if len(Index) > 2:#需要分段
            for index,i in enumerate(Index):
                if index < len(Index)-1:
                    tem1 = data.iloc[i:Index[index+1],]
                    tem2 = acc_cal(tem1)
                    result.append(tem2)
        else:
            tem3 = acc_cal(data)
            result.append(tem3)
    result = pd.concat(result,ignore_index=True)
    return result

def data_match(result,need_traj):
    '''
    为冲突数据匹配速度，加速度
    '''
    tem1 = need_traj[['timestamp','sog','acc','MMSI','COG','length']]
    tem2 = need_traj[['timestamp','sog','acc','MMSI','COG']]
    tem1 = tem1.rename(columns={'MMSI':'oM','sog':'oSog','acc':'oAcc',\
                                'COG':'oCog','length':'olen'})
    tem2 = tem2.rename(columns={'MMSI':'tM','sog':'tSog','acc':'tAcc',\
                                'COG':'tCog'})
    tem3 = pd.merge(result,tem1)
    tem4 = pd.merge(tem3,tem2)
    tem4.sort_values(by='timestamp',ascending=True,inplace=True)
    return tem4

#def ndata_match(df):
#    tem1 = new_maxB[['RB(r)','Lrk','maxDis']]
#    tem2 = tem1.rename(columns={'RB(r)':'newRB(r)','maxDis':'nMaxDis'})
#    tem3 = new_minB[['RB(r)','Lrk','minDis']]
#    tem4 = tem3.rename(columns={'RB(r)':'newRB(r)','minDis':'nMinDis'})
#    result = pd.merge(df,tem2)
#    result = pd.merge(result,tem4)
#    return result

def seCon_cal(df1):
    '''
    计算冲突严重度
    df1数据详解：
        (1)Dis: 两船之间相对距离
        (2)Lrk: 本船的长度等级
        (3)Rad: 他船相对于本船的相对方位(弧度)
        (4)RB: 他船相对于本船的相对方位(角度)
        (5)ID: oM*10+tM*0.1
        (6)RB(r): RB四舍五入
        (7)deCon: 冲突程度
        (8)oDiApp/tDiApp: 本船或他船在时间deltaT内行驶的距离
        (9)Cog,Sog,Rad: 航向，航速，航向（弧度）
        (10)RaApp: 接近率
        (11)seCon: 冲突严重度
    '''
    deltaT=30
    df1['RD'] = df1.apply(lambda x:x['Dis']*x['olen'],axis=1)
    df1['oDiApp'] = df1.apply(lambda x:x['oSog']*deltaT+0.5*x['oAcc']*\
       (deltaT**2),axis=1)#注意：速度单位选择m/s
    df1['tDiApp'] = df1.apply(lambda x:x['tSog']*deltaT+0.5*x['tAcc']*\
       (deltaT**2),axis=1)#注意：速度单位选择m/s
    df1['oRad'] = df1.apply(lambda x:math.radians(x['oCog']),axis=1)
    df1['tRad'] = df1.apply(lambda x:math.radians(x['tCog']),axis=1)
    df1['odx'] = df1.apply(lambda x:x['oDiApp']*math.sin(x['oRad']),axis=1)
    df1['ody'] = df1.apply(lambda x:x['oDiApp']*math.cos(x['oRad']),axis=1)
    df1['tdx'] = df1.apply(lambda x:x['tDiApp']*math.sin(x['tRad']),axis=1)
    df1['tdy'] = df1.apply(lambda x:x['tDiApp']*math.cos(x['tRad']),axis=1)
    #
    df1['oDx'] = df1.apply(lambda x:x['RD']*math.sin(x['Rad']),axis=1)
    df1['oDy'] = df1.apply(lambda x:x['RD']*math.cos(x['Rad']),axis=1)
    df1['newRD'] = df1.apply(lambda x:pow((pow((x['oDx']+x['tdx']-x['odx']),\
       2)+pow((x['oDy']+x['tdy']-x['ody']),2)),0.5),axis=1)
    df1['RaApp'] = df1.apply(lambda x:(x['RD']-x['newRD'])/x['RD'],axis=1)
    seCons = []
    for row in df1.itertuples():
        if row.deCon == 0:
            seCon = 0
            seCons.append(seCon)
        elif row.deCon < 1:
            seCon = pow(row.deCon,1-row.RaApp)
            seCons.append(seCon)
        else:
            seCon = pow(row.deCon,1+row.RaApp)
            seCons.append(seCon)
    df1['seCon'] = seCons
    return df1

#%%
def seCon_plot(data):
    '''
    画图：SeCon,RaApp,DeCon的变化趋势
    合适冲突的ID
    22: (1)4175721612.6
    20: (1)4176001409.0
    
    '''
    data.reset_index(drop=True,inplace=True)
    import time
    from datetime import datetime
    tem = data.copy()
    #该时间为地区时
    tem['time'] = tem['timestamp'].apply(lambda x:time.\
        strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    import matplotlib.pyplot as plt 
    from matplotlib import gridspec
    #import matplotlib.dates as mdate
    figsize = 11,14
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Conflict analysis',fontsize=22)
    x1 = tem.time.values
    x1= [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in x1]
    y1 = tem.seCon.values
    y2 = tem.deCon.values
    y3 = tem.RaApp.values
    gs = gridspec.GridSpec(2,1,height_ratios=[3, 1]) 
    ax1 = plt.subplot(gs[0])
    ax1 = plt.gca()   
    #ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))   
    #plt.xticks(pd.date_range(tem.loc[0,'time'],tem.loc[len(data)-1,'time'],freq='20S'))
    ax1.plot(x1,y1,linewidth=2.0,label='Conflict Severity')
    ax1.plot(x1,y2,linewidth=2.0,label='Conflict Degree')
    plt.legend()
    plt.xlabel('Time',Fontsize=18)
    plt.ylabel('Conflict',Fontsize=18)
    ax2 = plt.subplot(gs[1])
    ax2 = plt.gca()    
    ax2.plot(x1,y3,linewidth=1.0)
    plt.xlabel('Time',Fontsize=18)
    plt.ylabel('RaApp',Fontsize=18)
#%%
def con_output(data):
    '''
    一段冲突中需要输出的结果
    '''
    oM = data.oM.values[0]
    tM = data.tM.values[0]
    ID = oM*10+tM*0.1
    maxSev = data.seCon.max()
    maxDeg = data.deCon.max()
    Time1 = data[data['seCon'].isin([maxSev])].timestamp.values[0]
    Time2 = data[data['deCon'].isin([maxDeg])].timestamp.values[0]
    RaApp1 = data[data['seCon'].isin([maxSev])].RaApp.values[0]
    RaApp2 = data[data['deCon'].isin([maxDeg])].RaApp.values[0]
    time = data.timestamp.tolist()
    duration = max(time)-min(time)
    from scipy import integrate
    x = [ a -time[0] for a in time]
    y1 = data.seCon.tolist()
    y2 = data.deCon.tolist()
    Area1 = integrate.trapz(y1, x)
    Area2 = integrate.trapz(y2, x)
    if duration == 0:
        meanSev = np.nan
        meanDeg = np.nan
    else:
        meanSev = Area1/duration
        meanDeg = Area2/duration
    result = (oM,tM,ID,maxSev,maxDeg,Time1,Time2,duration,meanSev,meanDeg,RaApp1,RaApp2)
    return result

def end_cal(df1):
    '''
    计算maxSev,meanSev,duration
    '''
    con_count = 0
    Result = []
    new_df = []
    for group in df1.groupby('ID'):
        if group[1].shape[0] > 1:
            timestamp = group[1].timestamp.values
            Index = []
            for index,time in enumerate(timestamp):
                if index == 0:
                    Index.append(index)
                else:
                    diff = time - timestamp[index-1]
                    if diff > 300:#间隔超过五分钟，则认为冲突分段
                        Index.append(index)
            Index.append(index)
            if len(Index) ==2:
                data = group[1]
                result = con_output(data)
                Result.append(result)
                con_count = con_count + 1
                data1 = data.copy()
                data1['count'] = con_count
                new_df.append(data1)
            else:
                for j,idx in enumerate(Index):
                    if idx > 0:
                        data = group[1][Index[j-1]:idx]
                        if data.shape[0]>1:
                            result = con_output(data)
                            Result.append(result)
                            con_count = con_count + 1
                            data1 = data.copy()
                            data1['count'] = con_count
                            new_df.append(data1)
    new_DF = pd.concat(new_df,ignore_index=False)    
    Result = pd.DataFrame(Result,columns=['MMSI','tM','ID','maxSev','maxDeg','Time1','Time2','duration','meanSev','meanDeg','RaApp1','RaApp2'])
    #Result = pd.merge(Result,new_traj)
    Result = Result[~Result['duration'].isin([0])]
    return Result,new_DF
#%%
bounFile1 = 'D:/结果/boundary/max_boun.csv'
bounFile2 = 'D:/结果/boundary/min_boun.csv'
dataFile1 = 'D:/结果/20181016Data.csv'
trajFile1 = 'D:/论文数据/20181016(2s)processed.csv'
outputFile1 = 'D:/结果/20181016result.csv'
outputFile2 = 'D:/结果/20181016conf_data.csv'
maxBoun = pd.read_csv(bounFile1,encoding='gbk',engine='python')
minBoun = pd.read_csv(bounFile2,encoding='gbk',engine='python')
df = pd.read_csv(dataFile1,encoding='gbk',engine='python')
traj = pd.read_csv(trajFile1,encoding='gbk',engine='python')
Lrk = maxBoun.Lrk.unique()
#%%
new_maxB, new_minB = boun(maxBoun,minBoun)
p1,p2,p3,p4 = path_boun(new_maxB)
new_df = azi2azi(df)
new_df = len_cut(new_df)
new_df = add_state(new_df)
result = deal_df(new_df)
con_List,count_ship = con_MMSI(result)
new_traj = needed_traj(traj,con_List)
need_traj = acc_merge(new_traj)
df1 = data_match(result,need_traj)
df2 = seCon_cal(df1)
Result,new_DF = end_cal(df2)
#%%
data=new_DF[new_DF['count'].isin([5])]
seCon_plot(data)
#%%
Result.to_csv(outputFile1, sep=',', header=True,index=0)
new_DF.to_csv(outputFile2, sep=',', header=True,index=0)
