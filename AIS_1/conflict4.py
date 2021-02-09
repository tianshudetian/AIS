# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:49:59 2020

@author: 24592
"""

import pandas as pd
import scipy.interpolate as spi
from matplotlib.path import Path
#import matplotlib.pyplot as plt
import numpy as np
import math
#import time
#import datetime
#import seaborn as sns
from math import radians, cos, sin, asin, sqrt
#%%
def traj_filter(data):
    '''
        根据研究范围过滤船舶数据
    '''
    minlat = 29.8233333333
    maxlat = 29.9460166663
    minlon = 122.10
    maxlon = 122.25
    data = data.loc[(data.lon>=minlon)&(data.lat>=minlat)&(data.lon<=maxlon)&(data.lat<=maxlat)]
    return data

def ra2de(data,Lrk):
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

def new_boun(data,Lrk):
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

def boun(data1,data2,Lrk):
    '''
        生成最后所需的边界数据
    '''
    tem1 = ra2de(data1,Lrk)
    new_maxB = new_boun(tem1,Lrk)
    new_maxB = dxy2dis(new_maxB)
    new_maxB.rename(columns={'Dis':'maxDis','RB':'RB(r)'},inplace=True)
    tem2 = ra2de(data2,Lrk)
    new_minB = new_boun(tem2,Lrk)
    new_minB = dxy2dis(new_minB)
    new_minB.rename(columns={'Dis':'minDis','RB':'RB(r)'},inplace=True)
    return new_maxB,new_minB

def path_boun(data,Lrk):
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


def tm2om(df):
    '''
        RD:相对距离(km) RB：相对方位 len:船长 oM:本船MMSI tM:他船MMSI COG：航向 SOG：航速
        timestamp:时间戳
    '''
    df1 = df[['RD','oRB','olen','oM','tM','oCOG','oSOG','timestamp']]
    df2 = df[['RD','tRB','tlen','oM','tM','tCOG','tSOG','timestamp']]
    df3 = df1.copy()
    df4 = df2.copy()
    df3.rename(columns={'oRB':'RB','olen':'len','oCOG':'COG','oSOG':'SOG'},inplace=True)
    df4.rename(columns={'tRB':'RB','tlen':'len','tCOG':'COG','tSOG':'SOG','oM':'MMSI1','tM':'MMSI2'},inplace=True)
    df4.rename(columns={'MMSI1':'tM','MMSI2':'oM'},inplace=True)
    df5 = pd.concat([df3,df4],sort=True)
    df5.sort_values(by='timestamp',inplace=True)
    df5.reset_index(drop=True)
    return df5

def azi2azi(df):
    '''
        将相对方位RB转换为相对方位rb
        RB：他船经纬度点与本船经纬度点之间的相对方位
        rb：他船与本船之间的相对方位，即他船所在方位与本船航向之间的夹角
        Rad:rb转弧度
        Dis:相对距离与本船船长的比值，m/m,无量纲
    '''
    df['rb'] = df.apply(lambda x:x['RB']-x['COG'],axis=1)
    df['Rad'] = df.apply(lambda x:math.radians(x['rb']),axis=1)
    df['Dis'] = df.apply(lambda x:1000*x['RD']/x['len'],axis=1)
    df['dx'] = df.apply(lambda x:x['Dis']*math.sin(x['Rad']),axis=1)
    df['dy'] = df.apply(lambda x:x['Dis']*math.cos(x['Rad']),axis=1)
    return df

def len_cut(df):
    '''
        长度切割
    '''
    lenBins = [50,100,200,300,10000]
    lenLabels = ['1','2','3','4']
    df['Lrk'] = pd.cut(df['len'],bins=lenBins,labels=lenLabels ,include_lowest=True)
    df.dropna(axis=0,how='any',inplace=True)
    return df

def state_ana(Lrk,point,p1,p2,p3,p4):
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

def add_state(df,p1,p2,p3,p4):
    '''
        为df添加状态
    '''
    States = []
    for row in df.itertuples():
        Point = (row.dx,row.dy)
        State = state_ana(row.Lrk,Point,p1,p2,p3,p4)
        States.append(State)
    df['State'] =States
    return df

def deal_df(df,new_maxB,new_minB):
    '''
        冲突数据提取与边界匹配
    '''
    data = df[df['State'].isin([1])]
    tem = data.copy()
    tem['RB(r)'] = tem['rb'].apply(lambda x:round(x))#相对方位四舍五入
    a = tem.loc[tem['RB(r)']<=0]
    b = a.copy()
    b['RB(r)'] = b['RB(r)'].apply(lambda x:x+360)
    b['RB(r)']=b['RB(r)'].replace([360],[0])
    c = tem.loc[tem['RB(r)']>=1e-06]
    d = pd.concat([b,c])
    d['Lrk'] = d.Lrk.astype('int')
    tem1=new_maxB[['RB(r)','Lrk','maxDis']]
    tem2=new_minB[['RB(r)','Lrk','minDis']]
    tem3=pd.merge(d,tem1)
    tem3=pd.merge(tem3,tem2)
    tem3['deCon']=tem3.apply(lambda x:(x['Dis']-x['minDis'])/(x['maxDis']-x['minDis']),axis=1)
#    tem3['FaiMin']=tem3.apply(lambda x:-x['minDis']/(x['maxDis']-x['minDis']),axis=1)#计算当前最低的空间接近度
#    tem3.loc[tem3['deCon']<0,'deCon'] = 0#将计算小于0的值替换为0
    tem3.sort_values(by='timestamp',ascending=True,inplace=True)
    tem4 = tem3[['timestamp','RB','RD','rb','Rad','Dis','Lrk','deCon','COG','SOG','len','oM','tM']]
    tem5 = tem4.copy()
    tem5.rename(columns={'COG':'oCOG','SOG':'oSOG','len':'olen'},inplace=True)
    return tem5

def con_MMSI(df):
    '''
        输出参与冲突的船舶的MMSI号，及数量
    '''
    list1=df.oM.unique()
    list2=df.tM.unique()
    list3=np.concatenate((list1,list2),axis=0)
    list3 = pd.DataFrame(list3)
    list3.drop_duplicates(inplace=True)
    return list3

def needed_traj(df,List):
    '''
        根据参与冲突船舶的MMSI号，从轨迹数据中筛选出所需要的轨迹数据
    '''
    new_traj = []
    for row in List.itertuples():
        mmsi = row._1
        tem = df[df['MMSI'].isin([mmsi])]
        if tem.shape[0] >= 3:
            new_traj.append(tem)
    new_traj = pd.concat(new_traj,ignore_index=True)
    return new_traj

def data_match(data1,data2):
    '''
        为冲突数据匹配他船数据
        速度从kn转化为m/s
    '''
    tem1 = data2[['timestamp','MMSI','SOG','COG','length']]
    tem1 = tem1.rename(columns={'MMSI':'tM','SOG':'tSOG','COG':'tCOG','length':'tlen'})
    tem2 = pd.merge(data1,tem1)
    tem2['osog']=tem2.apply(lambda x:x['oSOG']*(1852/3600),axis=1)
    tem2['tsog']=tem2.apply(lambda x:x['tSOG']*(1852/3600),axis=1)
    tem2.drop(['oSOG','tSOG'],axis=1,inplace=True)
    return tem2

#def con_output(data):
#    '''
#    一段冲突中需要输出的结果
#    '''
#    oM = data.oM.values[0]
#    tM = data.tM.values[0]
#    maxDeg = data.deCon.min()###根据公式求最小，接近度最小，冲突程度越高
#    Time1 = data[data['deCon'].isin([maxDeg])].timestamp.values[0]
#    result = (oM,tM,maxDeg,Time1)
#    return result

def end_cal(df):
    '''
        计算
    '''
    data = df.copy()
    data['ID'] = data.apply(lambda x:x['oM']*x['tM'],axis=1)
    data.rename(columns={'oCOG':'ocog','tCOG':'tcog'},inplace=True)
    Timestamp = data['timestamp'].unique()
    delta_t = 20#20sec#指定时间间隔
    result = []
    failCount = 0
    for time in Timestamp:
        tem1 = data[data['timestamp'].isin([time])]
        ID = tem1['ID'].unique()
        for Id in ID:
            tem2 = tem1[tem1['ID'].isin([Id])]
            mmsi1 = tem2['oM'].iloc[0]
            mmsi2 = tem2['tM'].iloc[0]
            osog = tem2['osog'].iloc[0]
            tsog = tem2['tsog'].iloc[0]
            ocog = tem2['ocog'].iloc[0]
            tcog = tem2['tcog'].iloc[0]
            Az = tem2['RB'].iloc[0]
            d1 = tem2['RD'].iloc[0]*1000
            d2 = 0.5144444*osog*delta_t
            d3 = 0.5144444*tsog*delta_t
            #横向距离
            if (ocog > 180.0 and Az > 180.0) or (ocog < 180.0 and Az < 180.0):
                a = 1
            else:
                a = -1
            if (tcog > 180.0 and Az > 180.0) or (tcog < 180.0 and Az < 180.0):
                b = -1
            else:
                b = 1
            dd1 = d1*abs(sin(radians(Az))) + a*d2*abs(sin(radians(ocog))) +b*d3*abs(sin(radians(tcog)))
            #纵向距离
            dd2 = d1*abs(cos(radians(Az))) + d2*abs(cos(radians(ocog))) - d3*abs(cos(radians(tcog)))
            d = pow((dd1**2+dd2**2),0.5)
            v = (d1-d)/d
            alpha = 1.0
            beta = -0.3
            try:
                if tem2.shape[0] == 2:
                    Sev = []
                    for row in tem2.itertuples():
                        if row.deCon > beta:
                            Severity = (1/(row.deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
                            Sev.append(Severity)
                    res = (time,Id,mmsi1,mmsi2,Sev[0],Sev[1])
                elif tem2.shape[0] == 1:
                    deCon = tem2['deCon'].iloc[0]
                    if deCon > beta:
                        Severity = (1/(deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
                    res = (time,Id,mmsi1,mmsi2,Severity,0)
                result.append(res)
            except:
                failCount  = failCount + 1
    Result = pd.DataFrame(result)
    Result.set_axis(['time','ID','mmsi1','mmsi2','sev1','sev2'],axis='columns',inplace=True)
    Result['Sev'] = Result.apply(lambda x:pow((x['sev1']**2+x['sev2']**2),0.5),axis=1)
    Result.sort_values(by='time',inplace=True)
    Res = []
    for group in Result.groupby('ID'):
        tem3 = group[1].copy()
        maxSev = tem3['Sev'].max()
        T = tem3[tem3['Sev'].isin([maxSev])].time.values[0]
        T1 = tem3['time'].iloc[0]
        T3 = tem3['time'].iloc[-1]
        if (T-T1 != 0) and (-T+T3 != 0):
            R1 = (maxSev - tem3['Sev'].iloc[0])/(T-T1)
            R2 = (-maxSev + tem3['Sev'].iloc[-1])/(-T+T3)
            res = (T,maxSev,R1,R2,T1,T3)
            Res.append(res)
    RES = pd.DataFrame(Res)
    RES.set_axis(['time','maxSev','R1','R2','T1','T3'],axis='columns',inplace=True)
    RES = RES.loc[(RES['maxSev']<=4.075)]
    return Result,RES

def CPA(df):
    '''
    (r1,r2)为本船的速度向量Vo
    (r3,r4)为他船的速度向量Vt
    (r5,r6)为相对速度向量中的值V=Vt-Vo
    r7为相对速度的大小
    (r8,r9)为从本船到他船的相对距离，需求负
    r10为(r5,r6)与(r8,r9)两个向量的点乘
    r11为两船间的相对距离
    r12为相对速度（向量：V=Vt-Vo）与两船相对方位之间(向量：两个质点间的相对方位，并不是他船相对于本船的相对方位)的夹角
    r13为DCPA，RD*sin(0)
    r14为TCPA,RD*cos(0)
    '''
    data = df.copy()
    a = pd.DataFrame(columns=['r1'])
    a['r1'] = data.apply(lambda x:x['osog']*math.sin(math.radians(x['oCOG'])),axis=1)
    a['r2'] = data.apply(lambda x:x['osog']*math.cos(math.radians(x['oCOG'])),axis=1)
    a['r3'] = data.apply(lambda x:x['tsog']*math.sin(math.radians(x['tCOG'])),axis=1)
    a['r4'] = data.apply(lambda x:x['tsog']*math.cos(math.radians(x['tCOG'])),axis=1)
    a['r5'] = a.apply(lambda x:x['r3']-x['r1'],axis=1)
    a['r6'] = a.apply(lambda x:x['r4']-x['r2'],axis=1)
    a['r7'] = a.apply(lambda x:pow((x['r5']**2+x['r6']**2),0.5),axis=1)#相对速度的大小
    a['r8'] = data.apply(lambda x:(-1)*x['RD']*1000*math.sin(math.radians(x['RB'])),axis=1)
    a['r9'] = data.apply(lambda x:(-1)*x['RD']*1000*math.cos(math.radians(x['RB'])),axis=1)
    a['r10'] = a.apply(lambda x:np.dot([x['r5'],x['r6']],[x['r8'],x['r9']]),axis=1)
    a['r11'] = data['RD']
    a['r12'] = a.apply(lambda x:math.acos((x['r10']/(x['r7']*x['r11']*1000))),axis=1)
    tem = data.copy()
    tem['DCPA(m)'] = a.apply(lambda x:math.sin(x['r12'])*x['r11']*1000,axis=1)
    tem['TCPA(s)'] = a.apply(lambda x:1000*x['r11']*math.cos(x['r12'])/x['r7'],axis=1)
    tem['RD(m)'] = tem['RD'].apply(lambda x:x*1000)
    return tem


#%%
print("程序开始运行")
bounFile1 = 'D:/结果/boundary/max_boun.csv'
bounFile2 = 'D:/结果/boundary/min_boun.csv'
dataFile1 = 'D:/结果/20181001Data.csv'
trajFile1 = 'D:/论文数据/20181001(2s)processed.csv'
outputFile1 = 'D:/11.24结果/20181001result.csv'
outputFile2 = 'D:/11.24结果/20181001res.csv'
outputFile3 = 'D:/11.24结果/20181001df6.csv'
maxBoun = pd.read_csv(bounFile1,encoding='gbk',engine='python')
minBoun = pd.read_csv(bounFile2,encoding='gbk',engine='python')
df = pd.read_csv(dataFile1,encoding='gbk',engine='python')
traj = pd.read_csv(trajFile1,encoding='gbk',engine='python')
print("数据读取完毕")
print("边界数据处理开始")
Lrk = maxBoun.Lrk.unique()
new_maxB, new_minB = boun(maxBoun,minBoun,Lrk)
p1,p2,p3,p4 = path_boun(new_maxB,Lrk)
print("边界数据处理完毕")
print("开始提取冲突")
df1 = tm2om(df)
df2 = azi2azi(df1)
df3 = len_cut(df2)
df4 = add_state(df3,p1,p2,p3,p4)
df5 = deal_df(df4,new_maxB,new_minB)
List = con_MMSI(df5)
print("冲突提取完毕")
print("开始处理轨迹")
traj = traj_filter(traj)
new_traj = needed_traj(traj,List)
print("轨迹数据提取完毕")
print("开始计算结果")
df6 = data_match(df5,new_traj)
Result,RES = end_cal(df6)
#Result.to_csv(outputFile1, sep=',', header=True,index=0)
#RES.to_csv(outputFile2, sep=',', header=True,index=0)
#df6.to_csv(outputFile3, sep=',', header=True,index=0)
print("结果保存完毕")   