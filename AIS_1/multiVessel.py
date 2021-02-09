# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:09:28 2020

@author: dn4
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import radians,sin,cos,degrees,sqrt,asin,atan2
import scipy.interpolate as spi
import numpy as np
from matplotlib.pyplot import quiver
from matplotlib.path import Path
import time
import datetime

def rd_cal(data):
    r = 6371.393
    tem1 = data.copy()
    tem1['lat']=data.apply(lambda x:radians(x['lat']),axis=1)
    tem1['lon']=data.apply(lambda x:radians(x['lon']),axis=1)
    tem2 = []
    for row in tem1.itertuples():
        Set=tem1[tem1['mmsi']!=row.mmsi]
        result = pd.DataFrame(columns=['Rd'])#设置空DataFrame
        result['Rd']=Set.apply(lambda x:2*r*asin(sqrt(sin((x['lat']-row.lat)/2)**2+cos(row.lat)*cos(x['lat'])*sin((x['lon']-row.lon)/2)**2)),axis=1)
        result['Az']=Set.apply(lambda x:(degrees(atan2(sin(x['lon']-row.lon)*cos(x['lat']),cos(row.lat)*sin(x['lat'])-sin(row.lat)*cos(x['lat'])*cos(x['lon']-row.lon)))+360)%360,axis=1)
        result['om']=row.mmsi
        result['olen']=row.len#需保存row中船舶的长度
        result['cog']=row.cog
        tm=Set.pop('mmsi')#选出ts的MMSI
        result.insert(5,'tm',tm)
        tem2.append(result)
    Result = pd.concat(tem2,ignore_index=True)
    return Result

def ra2de(data,Lrk):
    '''
        (1)弧度转度数，使用math.degrees转化的度数有负值，不符合插值需求
        (2)因数据构成原型，所以最后一个方位，虽然是正数，却需要+360°
    '''
    data['RB'] = data.apply(lambda x:degrees(x['Rad']),axis=1)
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
    data['Dis'] = data.apply(lambda x:sqrt(pow(x['dx'],2)+pow(x['dy'],2)),axis=1)
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

def len_cut(df):
    '''
        长度切割
    '''
    lenBins = [50,100,200,300,10000]
    lenLabels = ['1','2','3','4']
    df['Lrk'] = pd.cut(df['olen'],bins=lenBins,labels=lenLabels ,include_lowest=True)
    df.dropna(axis=0,how='any',inplace=True)
    return df

def DeCon(df):
    tem = df.copy()
    tem['RB(r)'] = tem['rad'].apply(lambda x:round(degrees(x)))
    tem['Dis']=tem.apply(lambda x:1000*x['Rd']/x['olen'],axis=1)
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
    return tem3

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

def SeCon(om,tm,deCon,t,data):
    traj = data.copy()
    alpha = 1.0
    beta = -0.3
    trajOM = traj.loc[(traj.mmsi==om)]
    trajTM = traj.loc[(traj.mmsi==tm)]
    d1 = distance(trajOM,trajTM,t)
    t1 = t - 10
    d2 = distance(trajOM,trajTM,t1)
    v = (d1-d2)/d2
    Severity = (1/(deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
    return Severity

#def seCon(data1,data2):
#    df = data1.copy()
#    data = data2.copy()
#    seCon = []
#    for row in df.itertuples():
#        delta_t = 20#20sec
#        ocog = data[data['mmsi']==row.om].cog
#        osog = data[data['mmsi']==row.om].sog
#        tcog = data[data['mmsi']==row.tm].cog
#        tsog = data[data['mmsi']==row.tm].sog
#        d1 = row.Rd*1000
#        d2 = 0.5144444*list(osog)[0]*delta_t
#        d3 = 0.5144444*list(tsog)[0]*delta_t
#        #横向距离
#        if (list(ocog)[0] > 180.0 and row.Az > 180.0) or (list(ocog)[0] < 180.0 and row.Az < 180.0):
#            a = 1
#        else:
#            a = -1
#        if (list(tcog)[0] > 180.0 and row.Az > 180.0) or (list(tcog)[0] < 180.0 and row.Az < 180.0):
#            b = -1
#        else:
#            b = 1
#        dd1 = d1*abs(sin(radians(row.Az))) + a*d2*abs(sin(radians(list(ocog)[0]))) +b*d3*abs(sin(radians(list(tcog)[0])))
#        #纵向距离
#        dd2 = d1*abs(cos(radians(row.Az))) + d2*abs(cos(radians(list(ocog)[0]))) - d3*abs(cos(radians(list(tcog)[0])))
#        d = pow((dd1**2+dd2**2),0.5)
#        v = (d1-d)/d
#        alpha = 1.0
#        beta = -0.3
#        Severity = (1/(row.deCon-beta))**(1-alpha*v)-(1/(1-beta))**(1-alpha*v)
#        seCon.append(Severity)
#    df['seCon'] = seCon
#    return df
#%%
def plot_con(data1,data2,data3):
    df1=data1.copy()
    df2=data2.copy()
    tem=data3.copy()
    figsize = 6,5
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    for row in df1.itertuples():
        olon = tem[tem.mmsi==row.om]['lon'].values[0]
        olat = tem[tem.mmsi==row.om]['lat'].values[0]
        tlon = tem[tem.mmsi==row.tm]['lon'].values[0]
        tlat = tem[tem.mmsi==row.tm]['lat'].values[0]
        dx = tlon - olon
        dy = tlat - olat
#            
        quiver(tlon,tlat,-dx,-dy, angles='xy', scale=1.03, scale_units='xy', width=0.008)
    for row in df2.itertuples():
        ax1.scatter(row.lon,row.lat,c='black')
        ax1.text(row.lon-0.001,row.lat-0.002,row.v,fontsize=17)
        
    ax1.set_xlim(122.205,122.23)
    ax1.set_ylim(29.815,29.84)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax1.get_xaxis().get_major_formatter().set_scientific(False)
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    ax1.set_xlabel('经度(°)',fontsize=17)
    ax1.set_ylabel('纬度(°)',fontsize=17)
    plt.tick_params(labelsize=17)
    plt.show()
    
def plot_pos(data):
    df = data.copy()
    maxV = df['sog'].max()
    figsize = 6,5
    fig = plt.figure(figsize=figsize)
    ax2 = fig.add_subplot(111)
    for row in df.itertuples():
        a = 0.005
        b = row.sog/maxV
        ax2.scatter(row.lon,row.lat,c='black')
        dx = a*b*sin(radians(row.cog))
        dy = a*b*cos(radians(row.cog))
        quiver(row.lon,row.lat,dx,dy, angles='xy', scale=1.0, scale_units='xy', width=0.008)
        ax2.text(row.lon-0.001,row.lat-0.002,row.v,fontsize=17)
        
    ax2.set_xlim(122.205,122.23)
    ax2.set_ylim(29.815,29.84)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax2.get_xaxis().get_major_formatter().set_scientific(False)
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False    
    
    ax2.set_xlabel('经度(°)',fontsize=17)
    ax2.set_ylabel('纬度(°)',fontsize=17) 
    plt.tick_params(labelsize=17)
    plt.show()
#%%

bounFile1 = 'D:/结果/boundary/max_boun.csv'
bounFile2 = 'D:/结果/boundary/min_boun.csv'
maxBoun = pd.read_csv(bounFile1,encoding='gbk',engine='python')
minBoun = pd.read_csv(bounFile2,encoding='gbk',engine='python')
Lrk = maxBoun.Lrk.unique()
new_maxB, new_minB = boun(maxBoun,minBoun,Lrk)
p1,p2,p3,p4 = path_boun(new_maxB,Lrk)
dataFile1 = r'D:/论文数据/processed2s/20181023(2s)processed.csv'
df = pd.read_csv(dataFile1,encoding='gbk',engine='python')
df.rename(columns={"COG":"cog","SOG":"sog","MMSI":"mmsi","length":"len"},inplace=True)
df = df.reindex(columns=['lon','lat','cog','len','sog','mmsi','timestamp'])
timestamp = df['timestamp'].unique()
#%%
tData = timestamp
#timestamp = pd.DataFrame(df['timestamp'].unique())
#tData = timestamp.sample(frac=0.5)[0].tolist()
result = []
for tttt in tData:
    data = df[df['timestamp'].isin([tttt])]
    if data.shape[0] >= 20:
        df1 = rd_cal(data)
        df1 = len_cut(df1)
        df1['rad']=df1.apply(lambda x:radians(x['Az']-x['cog']),axis=1)
        df1['dx']=df1.apply(lambda x:1000*x['Rd']*sin(x['rad'])/x['olen'],axis=1)
        df1['dy']=df1.apply(lambda x:1000*x['Rd']*cos(x['rad'])/x['olen'],axis=1)
        df1 = add_state(df1,p1,p2,p3,p4)
        df1 = df1.loc[(df1.State==1)]
        if df1.shape[0] >= 5:
            a = (tttt,df1.shape[0])
            result.append(a)
result=pd.DataFrame(result)
#%%
def Si_cal(t,data):
    time = t
    df = data.copy()
    b=df[df.timestamp.isin([time])]
    if b.shape[0] >= 2:
        b1 = rd_cal(b)
        b1 = len_cut(b1)
        b1['rad']=b1.apply(lambda x:radians(x['Az']-x['cog']),axis=1)
        b1['dx']=b1.apply(lambda x:1000*x['Rd']*sin(x['rad'])/x['olen'],axis=1)
        b1['dy']=b1.apply(lambda x:1000*x['Rd']*cos(x['rad'])/x['olen'],axis=1)
        b1 = add_state(b1,p1,p2,p3,p4)
        b1 = b1.loc[(b1.State==1)]
        if b1.shape[0] >= 1:
            b2 =DeCon(b1)
            ####
            S = []
            for row in b2.itertuples():
                om=row.om
                tm=row.tm
                deCon = row.deCon
                s = SeCon(om,tm,deCon,time,df)
                S.append(s)
            b2['seCon'] = S
            tem = df[df.timestamp.isin([time])]
            tem2 = []
            for row in b2.itertuples():
                tem7 = tem[tem.mmsi==row.om]
                olon = tem7['lon'].values[0]
                olat = tem7['lat'].values[0]
                sog = tem7['sog'].values[0]
                Lon = olon
                Lat = olat
                tem1 = (Lon,Lat,row.om,row.tm,row.olen,row.cog,sog,row.deCon,row.seCon,time)
                tem2.append(tem1)
            tem3 = pd.DataFrame(tem2)
            tem3.set_axis(['lon','lat','om','tm','olen','cog','sog','deCon','seCon','t'],axis='columns',inplace=True)
            oMs=tem3['om'].unique()
            tMs=tem3['tm'].unique()
            MMSIs=set(np.hstack((oMs,tMs)))
            R = []
            for mmsi in MMSIs:
                try:
                    S_in = tem3[tem3['om'].isin([mmsi])]['seCon'].sum()
                except:S_in = 0
                try:
                    S_out = tem3[tem3['tm'].isin([mmsi])]['seCon'].sum()
                except:S_out = 0
                Si = (S_in+S_out)/(len(MMSIs)-1)
                r = (mmsi,S_in,S_out,Si,time)
                R.append(r)
            Re = pd.DataFrame(R)
            return Re,tem3
#%%
t = 1540279579
tmin = 1540279075
tmax = 1540280171
m = [352401000,477001700,240950000]
#求mID
#m = Si_cal(t,df)['mmsi']
tem10 = df[df['mmsi']==m[0]]
tem11 = df[df['mmsi']==m[1]]
tem12 = df[df['mmsi']==m[2]]
tem13 = tem11.append([tem10,tem12])
#time = tem13['timestamp'].unique()
R = []
S = []
#for ti in time:
for ti in range(tmin,tmax,2):
    r,s = Si_cal(ti,tem13)
    try:
        r.shape[0] > -1
        R.append(r)
        S.append(s)
    except:
        a =1
RR = pd.concat(R)
SS = pd.concat(S)
RR.columns=['mmsi','Sin','Sout','Si','t']
RR.reset_index(drop=True,inplace=True)
SS.reset_index(drop=True,inplace=True)
#%%
figsize = 6,5
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
RR['time'] = RR['t'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
for mmsi in m:
    Rtem = RR[RR['mmsi']==mmsi]
    titem = Rtem['time'].values
    timetem = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').time() for d in titem]
    ax.plot(timetem,Rtem['Sin'])
    
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

ax.set_xlabel('时间',fontsize=17)
ax.set_ylabel('$S^{in}$',fontsize=17) 
plt.tick_params(labelsize=17)
plt.show()
#%%
figsize = 6,5
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
SS['time'] = SS['t'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
for mmsi in m:
    Stem = SS[SS['om']==mmsi]
    titem = Stem['time'].values
    timetem = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').time() for d in titem]
    ax.plot(timetem,Stem['deCon'])
    
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

ax.set_xlabel('时间',fontsize=17)
ax.set_ylabel('冲突严重度',fontsize=17) 
plt.tick_params(labelsize=17)
plt.show()

#ax.plot()
#m11 = []
#m21 = []
#for i in range(len(m)):
#    try:
#        m1 = m[i]+m[i+1]
#        try:
#            m2 = m[i]+m[i+2]
#        except:
#            m2 = 0
#        m21.append(m2)
#    except:
#        m1 = 0
#    m11.append(m1)
#mtem = np.hstack((m11,m21))
#mID = []
#for j in mtem:
#    if j != 0:
#        mID.append(j)
#%%
temp = df.head(100000)
Timestamp =  temp.timestamp.unique()
x = []
y = []
for tim in Timestamp:
    temp1 = temp[temp['timestamp']==tim]
    plt.scatter(temp1['lon'],temp1['lat'])
    plt.pause(0.1)