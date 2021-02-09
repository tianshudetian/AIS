# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:10:08 2020

@author: dn4
"""
import pandas as pd
import numpy as np
from math import radians, cos, sin, acos, asin, sqrt

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
    a['r1'] = data.apply(lambda x:x['osog']*sin(radians(x['oCOG'])),axis=1)
    a['r2'] = data.apply(lambda x:x['osog']*cos(radians(x['oCOG'])),axis=1)
    a['r3'] = data.apply(lambda x:x['tsog']*sin(radians(x['tCOG'])),axis=1)
    a['r4'] = data.apply(lambda x:x['tsog']*cos(radians(x['tCOG'])),axis=1)
    a['r5'] = a.apply(lambda x:x['r3']-x['r1'],axis=1)
    a['r6'] = a.apply(lambda x:x['r4']-x['r2'],axis=1)
    a['r7'] = a.apply(lambda x:pow((x['r5']**2+x['r6']**2),0.5),axis=1)#相对速度的大小
    a['r8'] = data.apply(lambda x:(-1)*x['RD']*1000*sin(radians(x['RB'])),axis=1)
    a['r9'] = data.apply(lambda x:(-1)*x['RD']*1000*cos(radians(x['RB'])),axis=1)
    a['r10'] = a.apply(lambda x:np.dot([x['r5'],x['r6']],[x['r8'],x['r9']]),axis=1)
    a['r11'] = data['RD']
    a['r12'] = a.apply(lambda x:acos((x['r10']/(x['r7']*x['r11']*1000))),axis=1)
    tem = data.copy()
    tem['DCPA(m)'] = a.apply(lambda x:sin(x['r12'])*x['r11']*1000,axis=1)
    tem['TCPA(s)'] = a.apply(lambda x:1000*x['r11']*cos(x['r12'])/x['r7'],axis=1)
    tem['RD(m)'] = tem['RD'].apply(lambda x:x*1000)
    return tem

def distance(trajOM,trajTM,t):
    r = 6371.393
    a = trajOM[trajOM.timestamp==t]
    lat1 = radians(a['lat'].values[0])
    lon1 = radians(a['lon'].values[0])
    b = trajTM[trajTM.timestamp==t]
    lat2 = radians(b['lat'].values[0])
    lon2 = radians(b['lon'].values[0])
    distance = 2*r*asin(sqrt(sin((lat2-lat1)/2)**2+cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2))
    return distance

def end_cal(df1,df2):
    '''
        计算
    '''
    data = df1.copy()
    traj = df2.copy()
    data['ID'] = data.apply(lambda x:x['oM']*x['tM'],axis=1)
    data.rename(columns={'oCOG':'ocog','tCOG':'tcog'},inplace=True)
    Timestamp = data['timestamp'].unique()
    delta_t = 10#20sec#指定时间间隔
    result = []
    for time in Timestamp:
        tem1 = data[data['timestamp'].isin([time])]
        ID = tem1['ID'].unique()
        for Id in ID:
            tem2 = tem1[tem1['ID'].isin([Id])]
            mmsi1 = tem2['oM'].iloc[0]
            mmsi2 = tem2['tM'].iloc[0]
            dcpa = tem2['DCPA(m)'].iloc[0]
            tcpa = tem2['TCPA(s)'].iloc[0]
            rd = tem2['RD(m)'].iloc[0]
            d1 = tem2['RD'].iloc[0]*1000
#            Az = tem2['RB'].iloc[0]
#            osog = tem2['osog'].iloc[0]
#            tsog = tem2['tsog'].iloc[0]
#            ocog = tem2['ocog'].iloc[0]
#            tcog = tem2['tcog'].iloc[0]
#            d2 = 0.5144444*osog*delta_t
#            d3 = 0.5144444*tsog*delta_t
            #横向距离
#            if (ocog > 180.0 and Az > 180.0) or (ocog < 180.0 and Az < 180.0):
#                a = 1
#            else:
#                a = -1
#            if (tcog > 180.0 and Az > 180.0) or (tcog < 180.0 and Az < 180.0):
#                b = -1
#            else:
#                b = 1
#            dd1 = d1*abs(sin(radians(Az))) + a*d2*abs(sin(radians(ocog))) +b*d3*abs(sin(radians(tcog)))
#            #纵向距离
#            dd2 = d1*abs(cos(radians(Az))) + d2*abs(cos(radians(ocog))) - d3*abs(cos(radians(tcog)))
#            d = pow((dd1**2+dd2**2),0.5)
            #
            trajOM = traj[traj['MMSI'].isin([mmsi1])]
            trajTM = traj[traj['MMSI'].isin([mmsi2])]
            t = time - delta_t
            try:
                d = (distance(trajOM,trajTM,t))*1000
            except:
                continue
            #
            v = (d1-d)/d
            alpha = 1.0
            beta = -0.3
            if tem2.shape[0] == 2:
                dec = []
                Sev = []
                for row in tem2.itertuples():
                    if row.deCon > beta:
                        Severity = (1/(row.deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
                        Sev.append(Severity)
                        dec.append(row.deCon)
                if len(dec) == 2:
                    res = (time,Id,mmsi1,mmsi2,dec[0],Sev[0],dec[1],Sev[1],v,d1,d,dcpa,tcpa,rd)
            elif tem2.shape[0] == 1:
                deCon = tem2['deCon'].iloc[0]
                if deCon > beta:
                    Severity = (1/(deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
                res = (time,Id,mmsi1,mmsi2,deCon,Severity,0,0,v,d1,d,dcpa,tcpa,rd)
            result.append(res)
    Result = pd.DataFrame(result)
    Result.set_axis(['time','ID','mmsi1','mmsi2','dec1','sev1','dec2','sev2','v','d1','d','DCPA(m)','TCPA(s)','RD(m)'],axis='columns',inplace=True)
    Result['Sev'] = Result.apply(lambda x:pow((x['sev1']**2+x['sev2']**2),0.5),axis=1)
    Result.sort_values(by='time',inplace=True)
    return Result

def traj_mmsi_new(traj,mmsi,tmin,tmax):
    data = traj.copy()
    tem_traj = data.loc[(data['MMSI']==mmsi)&(data['timestamp']>=tmin)&(data['timestamp']<=tmax)]
    newTime = np.arange(tmin,tmax+2,10)
    temData2 = []
    for newT in newTime:
        temData1 = tem_traj[tem_traj['timestamp'].isin([newT])]
        temData2.append(temData1)
    re = pd.concat(temData2,ignore_index=True)
    return re

def traj_mmsi(traj,mmsi,t1,t2):
    '''
        traj:轨迹数据集
        t1:初始时间
        t2:终止时间时间
    '''
    data = traj.copy()
    shipTraj = data.loc[(data['timestamp']>=t1)&(data['timestamp']<=t2)]
    return shipTraj

def minmaxT(data1,mmsi):
    temp = data1.copy()
    tem1 = temp[temp['mmsi1'].isin([mmsi])][['time','sev1','dec1','v','d1','d']]
    tem2 = temp[temp['mmsi2'].isin([mmsi])][['time','sev2','dec2','v','d1','d']]
    tem2.rename(columns = {'sev2':'sev1','dec2':'dec1'},inplace=True)
    tem3 = pd.concat([tem1,tem2],ignore_index=True)
    tem4 = tem3[tem3['sev1']>0]
    tem4.sort_values(by='time',inplace=True, ascending=True)
    tem4.reset_index(drop = True,inplace=True)
    tmin = min(tem4['time'])
    tmax = max(tem4['time'])
    sev = tem4.loc[:,['sev1']]
    return tmin,tmax,sev

def Time2StrTime(tmin,tmax):
    '''
        时间区间生成时间序列
    '''
    import time
    import datetime
    Times = np.arange(tmin,tmax+2,2)
    strTime = [(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))) for x in Times]
    Time = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').time() for d in strTime]
    return Times,Time

def lonlat(data,mmsi,t):
    '''
        查找经纬度
    '''
    traj = data.copy()
    a=traj.loc[(traj['MMSI']==mmsi)&(traj['timestamp']==t)]
    lon = a.iloc[:,2].values[0]
    lat = a.iloc[:,3].values[0]
    return lon,lat

def display(temp1,temp2):
    import matplotlib.pyplot as plt
    data = temp1.copy()
    traj = temp2.copy()
    MMSIs = list(set(data['mmsi1']))
    if len(MMSIs) == 2:
        mmsi1 = MMSIs[0]
        mmsi2 = MMSIs[1]
        t1min,t1max,sev1 = minmaxT(data,mmsi1)
        t2min,t2max,sev2 = minmaxT(data,mmsi2)
    else:
        mmsi1 = data['mmsi1'].unique()[0]
        mmsi2 = data['mmsi2'].unique()[0]
        time = data['time']
        t1min = time.min()
        t1max = time.max()
        sev1 = data['sev1']
        tem_mmsi21 = temp[['time','sev2']]
        tem_mmsi22 = tem_mmsi21.loc[(tem_mmsi21['sev2']>0)]
        sev2 = tem_mmsi22['sev2']
        t2min = tem_mmsi22['time'].min()
        t2max = tem_mmsi22['time'].max()
    delta_T = 120 #设置轨迹前后延长120s
    dd  = traj_mmsi(traj,mmsi2,t2min,t2max)
    print(mmsi1,mmsi2)
    if dd.shape[0] >= 1:
        tmin = min(t1min,t2min)
        tmax = min(t1max,t2max)
        tem_traj1 = traj_mmsi_new(traj,mmsi1,tmin-delta_T,tmax+delta_T)
        tem_traj2 = traj_mmsi_new(traj,mmsi2,tmin-delta_T,tmax+delta_T)
        tem1  = traj_mmsi(tem_traj1,mmsi1,t1min,t1max)
        tem2  = traj_mmsi(tem_traj2,mmsi2,t2min,t2max)
        tem11  = traj_mmsi(tem_traj1,mmsi1,tmin-delta_T,t1min-2)
        tem21  = traj_mmsi(tem_traj2,mmsi2,tmin-delta_T,t2min-2)
        tem12  = traj_mmsi(tem_traj1,mmsi1,t1max+2,tmax+delta_T)
        tem22  = traj_mmsi(tem_traj2,mmsi2,t2max+2,tmax+delta_T)
    else:
        tmin = t1min
        tmax = t1max
        tem_traj1 = traj_mmsi_new(traj,mmsi1,tmin-delta_T,tmax+delta_T)
        tem_traj2 = traj_mmsi_new(traj,mmsi2,tmin-delta_T,tmax+delta_T)
        tem1  = traj_mmsi(tem_traj1,mmsi1,t1min,t1max)
        tem2  = traj_mmsi(tem_traj2,mmsi2,t2min,t2max)
        tem11  = traj_mmsi(tem_traj1,mmsi1,tmin-delta_T,t1min-2)
        tem21  = traj_mmsi(tem_traj2,mmsi2,tmin-delta_T,t1min-2)
        tem12  = traj_mmsi(tem_traj1,mmsi1,t1max+2,tmax+delta_T)
        tem22  = traj_mmsi(tem_traj2,mmsi2,t1min,tmax+delta_T)
#        print()
        

    MMSI1 = str(mmsi1)
    MMSI2 = str(mmsi2)
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 16,
             }
#    try:
    X1,x1 = Time2StrTime(t1min,t1max)
    X2,x2 = Time2StrTime(t2min,t2max)
    T1 = x1[np.argmax(sev1.values)]
    t1 = X1[np.argmax(sev1.values)]
    T11 = x1[np.argmax(sev1.values)+1]#参考线时间
    T2 = x2[np.argmax(sev2.values)]
    t2 = X2[np.argmax(sev2.values)]
    T21 = x2[np.argmax(sev2.values)+1]
    X3,x3 = Time2StrTime(min(t1min,t2min),max(t1max,t2max))
    tempn = data.loc[(data['time']>=min(t1min,t2min))&(data['time']<=max(t1max,t2max))]
    RD = tempn['RD(m)']
    T3 = x3[np.argmin(RD.values)]
    t3 = X3[np.argmin(RD.values)]
    T31 = x3[np.argmin(RD.values)+1]
    ####

    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sValue = 16
    plt.scatter(tem1['lon'],tem1['lat'],marker='x',s=sValue,norm=0.07,alpha=0.8,color='#0066CC')
    plt.scatter(tem2['lon'],tem2['lat'],marker='+',s=sValue,norm=0.07,alpha=1,color='#FF9966')
    ax.legend((MMSI1[0:5]+'****',MMSI2[0:5]+'****'),loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    plt.scatter(tem11['lon'],tem11['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem12['lon'],tem12['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem21['lon'],tem21['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem22['lon'],tem22['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    lon11,lat11 = lonlat(traj,mmsi1,t1)
    lon21,lat21 = lonlat(traj,mmsi2,t1)
    ax.scatter(lon11,lat11,s=sValue,c='k',marker='o')
    ax.text(lon11+0.0005,lat11-0.0005,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax.scatter(lon21,lat21,s=sValue,c='k',marker='o')
    ax.text(lon21-0.0025,lat21-0.0015,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    lon13,lat13 = lonlat(traj,mmsi1,t3)
    lon23,lat23 = lonlat(traj,mmsi2,t3)
    ax.scatter(lon13,lat13,s=sValue,c='k',marker='o')
    ax.text(lon13,lat13+0.0003,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax.scatter(lon23,lat23,s=sValue,c='k',marker='o')
    ax.text(lon23-0.0026,lat23-0.0019,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')      
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel('Longitude [°/E]',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('Latitude [°/N]',fontsize=17,fontname = 'Times New Roman') 
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')
    plt.show()

    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    pt1 = ax.plot(x1,sev1,ls='--',c='#0066CC',label=MMSI1[0:5]+'****')
    pt2 = ax.plot(x2,sev2,ls='--',c='#99CC33',label=MMSI2[0:5]+'****')
    plt.grid(linestyle='--')
    ax.axvline(x=T1,color='#FF3333',ls=':')
    ax.text(T11,0.22,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    print('T1: '+str(T1))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel('Time',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('Confilct severity ',fontsize=17,fontname = 'Times New Roman') 
    
    ax1 = ax.twinx()
    pt3 = ax1.plot(x3,RD,color='#FF9966',linestyle='-.',linewidth=1.5,label='RD')
    ax1.axvline(x=T3,color='#FF3333',ls=':')
    ax.text(T31,0.22,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    print('T3: '+str(T3))
    ax1.set_ylabel('Relative distance [m]',fontsize=17,fontname = 'Times New Roman') 
    pts = pt1 + pt2
    labs = [pt.get_label() for pt in pts]
    ax.legend(pts,labs,loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    ax1.legend(loc='upper right',prop=font1)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)
    plt.show()
    
    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ln1 = ax.plot(x3,tempn['DCPA(m)'],ls='--',c='#0066CC',label = 'DCPA')
    ax.set_xlabel('Time',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('DCPA [m]',fontsize=17,fontname = 'Times New Roman')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.grid(linestyle='--')
    ax1 = ax.twinx()
    ln2 = ax1.plot(x3,tempn['TCPA(s)'],ls='--',c='#99CC33',label = 'TCPA')
    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]
    ax.legend(lns,labs,loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    ax1.set_ylabel('TCPA [s]',fontsize=17,fontname = 'Times New Roman') 
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)
    ax1.axvline(x=T1,color='#FF3333',ls=':')
    ax1.text(T11,8,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax1.axvline(x=T3,color='#FF3333',ls=':')
    T32 = x3[np.argmin(RD.values)+1]
    ax1.text(T32,8,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax1.axhline(y=0,color='#FF3333',ls=':')
    plt.show()
#    except:
#        print('ValueError: arange: cannot compute length')

    return sev1
#    except:
#        print('轨迹数据不足前后2分钟')
    
    
    
    
#%%
#inputFile1 = 'D:/11.24结果/df/20181023df6.csv'
trajFile1 = 'D:/论文数据/20181024(2s)processed.csv'
#outputFile = 'D:/11.24结果/20181023df6res.csv'
#df = pd.read_csv(inputFile1,encoding='gbk',engine='python')
traj = pd.read_csv(trajFile1,encoding='gbk',engine='python')
#df1 = CPA(df)
#df2 = end_cal(df1,traj)
#df2.to_csv(outputFile)
df2 = pd.read_csv('D:/11.24结果/20181024df6res.csv',encoding='gbk',engine='python')
IDs=df2['ID'].unique()
#%%
ID = IDs[50]
#24日:50
#23日:46
temp = df2[df2['ID'].isin([ID])]
a=display(temp,traj)


#%%
    X1,x1 = Time2StrTime(t1min,t1max)
    X2,x2 = Time2StrTime(t2min,t2max)
    T1 = x1[np.argmax(sev1.values)]
    t1 = X1[np.argmax(sev1.values)]
    T11 = x1[np.argmax(sev1.values)+1]#参考线时间
    T2 = x2[np.argmax(sev2.values)]
    t2 = X2[np.argmax(sev2.values)]
    T21 = x2[np.argmax(sev2.values)+1]
    X3,x3 = Time2StrTime(min(t1min,t2min),max(t1max,t2max))
    tempn = data.loc[(data['time']>=min(t1min,t2min))&(data['time']<=max(t1max,t2max))]
    RD = tempn['RD(m)']
    sev = tempn['Sev']
    T3 = x3[np.argmin(RD.values)]
    t3 = X3[np.argmin(RD.values)]
    T31 = x3[np.argmin(RD.values)+1]
    ####

    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sValue = 16
    plt.scatter(tem1['lon'],tem1['lat'],marker='x',s=sValue,norm=0.07,alpha=0.8,color='#0066CC')
    plt.scatter(tem2['lon'],tem2['lat'],marker='+',s=sValue,norm=0.07,alpha=1,color='#FF9966')
    ax.legend((MMSI1[0:5]+'****',MMSI2[0:5]+'****'),loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    plt.scatter(tem11['lon'],tem11['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem12['lon'],tem12['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem21['lon'],tem21['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    plt.scatter(tem22['lon'],tem22['lat'],color='#99CC33',marker='.',s=sValue,norm=0.02,alpha=1)
    lon11,lat11 = lonlat(traj,mmsi1,t1)
    lon21,lat21 = lonlat(traj,mmsi2,t1)
    ax.scatter(lon11,lat11,s=sValue,c='k',marker='o')
    ax.text(lon11,lat11,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax.scatter(lon21,lat21,s=sValue,c='k',marker='o')
    ax.text(lon21-0.0022,lat21+0.0002,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    lon12,lat12 = lonlat(traj,mmsi1,t2)
    lon22,lat22 = lonlat(traj,mmsi2,t2)
    ax.scatter(lon12,lat12,s=sValue,c='k',marker='o')
    ax.text(lon12-0.0002,lat12+0.0006,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax.scatter(lon22,lat22,s=sValue,c='k',marker='o')
    ax.text(lon22-0.002,lat22-0.0005,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')  
    lon13,lat13 = lonlat(traj,mmsi1,t3)
    lon23,lat23 = lonlat(traj,mmsi2,t3)
    ax.scatter(lon13,lat13,s=sValue,c='k',marker='o')
    ax.text(lon13,lat13+0.0001,'T3',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax.scatter(lon23,lat23,s=sValue,c='k',marker='o')
    ax.text(lon23-0.002,lat23,'T3',fontsize=16,fontfamily='Times New Roman',weight='normal')      
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel('Longitude [°/E]',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('Latitude [°/N]',fontsize=17,fontname = 'Times New Roman') 
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')
    plt.show()
    

    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    pt1 = ax.plot(x1,sev1,ls='--',c='#0066CC',label=MMSI1[0:5]+'****')
    pt2 = ax.plot(x2,sev2,ls='--',c='#99CC33',label=MMSI2[0:5]+'****')
#    pt3 = ax.plot(x3,sev,color='#FF9966',linestyle='-.',linewidth=1.5,label='sev')
    plt.grid(linestyle='--')
    ax.axvline(x=T1,color='#FF3333',ls=':')
    ax.text(T11,0.22,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    print('T1: '+str(T1))
    ax.axvline(x=T2,color='#FF3333',ls=':')
    ax.text(T21,0.22,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    print('T2: '+str(T2))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel('Time',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('Confilct severity ',fontsize=17,fontname = 'Times New Roman') 
    
    ax1 = ax.twinx()
    ax1.plot(x3,RD,color='#FF9966',linestyle='-.',linewidth=1.5,label='RD')
    ax1.axvline(x=T3,color='#FF3333',ls=':')
    ax.text(T31,0.22,'T3',fontsize=16,fontfamily='Times New Roman',weight='normal')
    print('T3: '+str(T3))
    ax1.set_ylabel('Relative distance [m]',fontsize=17,fontname = 'Times New Roman') 
    pts = pt1 + pt2
#    pts = pt1 + pt2 + pt3
    labs = [pt.get_label() for pt in pts]
    ax.legend(pts,labs,loc='lower left',prop=font1,ncol=3, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    ax1.legend(loc='upper right',prop=font1)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)
    plt.show()
    
    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ln1 = ax.plot(x3,tempn['DCPA(m)'],ls='--',c='#0066CC',label = 'DCPA')
    ax.set_xlabel('Time',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('DCPA [m]',fontsize=17,fontname = 'Times New Roman')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.grid(linestyle='--')
    ax1 = ax.twinx()
    ln2 = ax1.plot(x3,tempn['TCPA(s)'],ls='--',c='#99CC33',label = 'TCPA')
    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]
    ax.legend(lns,labs,loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
    ax1.set_ylabel('TCPA [s]',fontsize=17,fontname = 'Times New Roman') 
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)
    ax1.axvline(x=T1,color='#FF3333',ls=':')
    ax1.text(T11,8,'T1',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax1.axvline(x=T2,color='#FF3333',ls=':')
    ax1.text(T21,8,'T2',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax1.axvline(x=T3,color='#FF3333',ls=':')
    T32 = x3[np.argmin(RD.values)+1]
    ax1.text(T32,8,'T3',fontsize=16,fontfamily='Times New Roman',weight='normal')
    ax1.axhline(y=0,color='#FF3333',ls=':')
    plt.show()