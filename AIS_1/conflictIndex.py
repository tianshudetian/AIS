# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:20:52 2020

@author: dn4
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin

def read(fileName):
    df = []
    for info in os.listdir(fileName): 
        domain = os.path.abspath(fileName) #获取文件夹的路径
        info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
        data = pd.read_csv(info,encoding='gbk',engine='python') 
        df.append(data)
    df = pd.concat(df)
    return df

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
        if (T-T1 > 4) and (-T+T3 > 4):
            x = list(tem3['time'])-T1
            y = list(tem3['Sev'])
            area = np.trapz(y,x,dx=0.1)
            R1 = (maxSev - tem3['Sev'].iloc[0])/(T-T1)
            R2 = (-maxSev + tem3['Sev'].iloc[-1])/(-T+T3)
            res = (T,maxSev,R1,R2,T1,T3,area)
            Res.append(res)
    RES = pd.DataFrame(Res)
    RES.set_axis(['time','maxSev','R1','R2','T1','T3','Area'],axis='columns',inplace=True)
    RES = RES.loc[(RES['maxSev']<=4.075)]
    return RES
#r'D:/11.24结果/res'
#r'D:/11.24结果/res'
#%%
df=read(r'D:/11.24结果/df')
res=end_cal(df)
res['duration'] = res.apply(lambda x:x['T3']-x['T1'],axis=1)
res['ave'] = res.apply(lambda x:x['Area']/x['duration'],axis=1)
