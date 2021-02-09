# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:40:05 2019

@author: dn4

Purpose:display CSSD
    
file number: five
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import math
import seaborn as sns
#from scipy import interpolate

'''
运行代码前需检查下列内容：
(1)输入输出地址
(2)输出方位
(3)图片标签
(4)CSSD的取值范围
(5)核密度估计的带宽
(6)核密度估计所需的等差数列
'''
# =============================================================================
df_00 = pd.read_csv("D:/论文数据/201707CSSD.csv",encoding='gbk',engine='python')
df_01 = pd.read_csv("D:/论文数据/20181001CSSD.csv",encoding='gbk',engine='python')
df_02 = pd.read_csv("D:/论文数据/20181002CSSD.csv",encoding='gbk',engine='python')
df_03 = pd.read_csv("D:/论文数据/20181003CSSD.csv",encoding='gbk',engine='python')
df_04 = pd.read_csv("D:/论文数据/20181004CSSD.csv",encoding='gbk',engine='python')#文件中存在较多的异常值
df_06 = pd.read_csv("D:/论文数据/20181006CSSD.csv",encoding='gbk',engine='python')
df_07 = pd.read_csv("D:/论文数据/20181007CSSD.csv",encoding='gbk',engine='python')
df_16 = pd.read_csv("D:/论文数据/20181016CSSD.csv",encoding='gbk',engine='python')
df_17 = pd.read_csv("D:/论文数据/20181017CSSD.csv",encoding='gbk',engine='python')
df_18 = pd.read_csv("D:/论文数据/20181018CSSD.csv",encoding='gbk',engine='python')
df_19 = pd.read_csv("D:/论文数据/20181019CSSD.csv",encoding='gbk',engine='python')
df_20 = pd.read_csv("D:/论文数据/20181020CSSD.csv",encoding='gbk',engine='python')#文件中存在较多的异常值
df_21 = pd.read_csv("D:/论文数据/20181021CSSD.csv",encoding='gbk',engine='python')
df_22 = pd.read_csv("D:/论文数据/20181022CSSD.csv",encoding='gbk',engine='python')
df_23 = pd.read_csv("D:/论文数据/20181023CSSD.csv",encoding='gbk',engine='python')
df_24 = pd.read_csv("D:/论文数据/20181024CSSD.csv",encoding='gbk',engine='python')
df = pd.concat([df_00,df_01,df_02,df_03,df_04,df_06,df_07,df_16,df_17,df_18,df_19,df_20,df_21,df_22,df_23,df_24])
outputFile1 = 'C:/Users/dn4/Desktop/min_boun.csv'
outputFile2 = 'C:/Users/dn4/Desktop/max_boun.csv'
#%%初始处理
def df_deal(data1,lim):
    '''
    (1)计算CSSD，相对距离与本船船长的比值
    (2)方位转极坐标弧度，方位无需转换为极坐标方位，根据画图进行调整
    (3)CSSD分解
    (4)简单的距离筛选
    (5)距离分解
    (6)方位数据转换
    '''
    data = data1.copy()
    data['CSSD'] = data.apply(lambda x:x['rDistance']/x['length'],axis=1)
    data['Theta'] = data.apply(lambda x: math.radians(x['rBearing']),axis=1)
    data['dx'] = data.apply(lambda x:x['CSSD']*math.sin(x['Theta']),axis=1)
    data['dy'] = data.apply(lambda x:x['CSSD']*math.cos(x['Theta']),axis=1)
    data = data.loc[(data['dx']>=-lim) & (data['dx']<=lim) & (data['dy']>=-lim) & (data['dy']<=lim)] 
    data['ID'] = data.apply(lambda x:x['osMMSI']*10+x['tsMMSI']*0.1,axis=1)
    id1 = data[data.rDistance<40].ID.tolist()
    for i in id1:
        data=data[~data.ID.isin([i])]
    data['rdx'] = data.apply(lambda x:x['rDistance']*math.sin(x['Theta']),axis=1)
    data['rdy'] = data.apply(lambda x:x['rDistance']*math.cos(x['Theta']),axis=1)
    data['BRank']=data['BRank'].astype('int')
    return data
#画图前预调取
def pre_plot():
    plt.rcParams['font.sans-serif']=['SimHei']#中文支持
    plt.rcParams['axes.unicode_minus']=False#中文支持
#相对位置分布图
def rD_dis(data,remark,labels):
    figsize = 10,20
    fig = plt.figure(figsize=figsize)
    fig.suptitle('相对位置点分布'+'('+remark+')',fontsize=18)
    for group in data.groupby('lenRank'):    
        ax = plt.subplot(2,2,group[0])
        ax.scatter(group[1].rdx,group[1].rdy,marker='.', c='k', alpha=0.5,s=10)#s调整点的大小
        ax.set_title(labels[group[0]-1])
#画CSSD的分布图
def CSSD_dis(df,remark,labels):
    title_labels = ['a','b','c','d']
    figsize = 10,20
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
#    fig.suptitle('CSSD(L)分布'+'('+remark+')',fontsize=18)
    lrk = df['lenRank'].unique()
    for i in lrk:
        data = df[df['lenRank'].isin([i])]
        ax = plt.subplot(2,2,i)
        ax.scatter(data.dx,data.dy,marker='.', c='k', alpha=0.5,s=5)#s调整点的大小
        #ax.set_title(labels[group[0]-1])
#        ax.set_title('('+title_labels[i-1]+')'+labels[i-1],fontsize=14)
#        ax.set_ylabel("D/L",fontsize=20)
#        ax.set_xlabel("D/L",fontsize=20)
        plt.tick_params(labelsize=20)
    
#搜索最小值
def min_cssd(data):
    min_res = []
    for group in data.groupby('lenRank'):
        for group1 in group[1].groupby('BRank'):
            min_CSSD=group1[1][group1[1].CSSD.isin([group1[1].CSSD.min()])]
            min_res.append(min_CSSD)
    min_res = pd.concat(min_res,ignore_index=True)
    minDens = min_res[['CSSD','Theta','lenRank','BRank']]
    minDens = minDens.rename(columns={'CSSD':'minPos','Theta':'Rad','lenRank':'Lrk','BRank':'Brk'})#位置、方位等级、速度等级
    return minDens
#核密度估计，输出核密度最大点的位置
def kde_cssd(data,bandwidth,lim):
    #核密度估计所需的等差数列
    dlim= 0
    ulim= lim
    inval= 2000
    X_plot = np.linspace(dlim,ulim,inval)[:,np.newaxis]#创建等差数列
    maxDens = []
    for i in Lrk:
        df = data[data['lenRank']==i]
        for group in df.groupby('BRank'):
            X = group[1]['CSSD'].values
            kde = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(X.reshape(-1,1))
            log_dens = kde.score_samples(X_plot)#返回的是点x对应概率的log值，要使用exp求指数还原
            log_dens = log_dens.tolist()#数组转列表
            max_pos = log_dens.index(max(log_dens))*ulim/inval#返回概率密度最大所在的位置(L)             
            bou_pos = [max_pos,group[0],i]#边界位置，对应概率密度最大点及其方位
            maxDens.append(bou_pos)
    maxDens = pd.DataFrame(maxDens)
    maxDens = maxDens.rename(columns={0:'maxPos',1:'Brk',2:'Lrk'})#位置、方位等级、速度等级
    maxDens['Rad'] = maxDens.apply(lambda x: math.radians(x['Brk']),axis=1)
    return maxDens
#画领域边界极坐标
    
def polar_cssd(data1,data2,labels):
    '''
    data1为期望边界
    data2为禁止边界
    labels为船长等级
    '''
#    title_labels = ['a','b','c','d']
    outLim = math.ceil(data1['maxPos'].max())
#    fig.suptitle('Domain Boundary',fontsize=18)
    for i in Lrk:
        maxBoun = data1[data1['Lrk']==i]
        Tail_devourer1 = maxBoun[0:1]
        maxBoun=maxBoun.append(Tail_devourer1,ignore_index=True)
        minBoun = data2[data2['Lrk']==i]
        Tail_devourer2 = minBoun[0:1]
        minBoun=minBoun.append(Tail_devourer2,ignore_index=True)
        #
        figsize = 4,4
#        figsize = 6,5
        fig = plt.figure(figsize=figsize)
        sns.set_context("paper",font_scale=1.5)
        sns.set(style='ticks')
        ax = fig.add_subplot(111,projection='polar')
#        ax.set_title('('+title_labels[i-1]+')'+labels[i-1],fontsize=14)
        ax.plot(maxBoun.Rad, maxBoun.maxPos, c='b', alpha=1,lw=1.5)
        ax.plot(minBoun.Rad, minBoun.minPos, c='r', alpha=1,lw=1.5)
        ax.set_theta_direction('clockwise')#   逆时针
        ax.set_theta_zero_location('N')#   极坐标0°位置为正北
        ax.set_thetagrids(np.arange(0.0,360.0,45.0))#  角度网格线显示
        ax.set_rgrids(np.arange(0.0,outLim,2.0))#    极径网格线显示
        plt.tick_params(labelsize=20)
#        ax.set_rlabel_position('90')# 极径标签位置
        #ax.set_rlim()#    极径范围设置
#生成所需的方位
def bea_choo():
    Bearings = []
    for bearing in range(45,360,90):
        Bearings.append(bearing)
    return Bearings
def cssd_clear(data,th,ca_th,dx_th,dy_th):
    '''
    (1)th: CSSD搜索范围的阈值
    (2)ca_th: CSSD搜素范围内允许容纳值数量的阈值
    (3)dx_th: 空间选取范围的阈值，建议其阈值大小与禁止边界接近
    (4)dy_th: 空间选取范围的阈值，建议其阈值大小与禁止边界接近
    '''
    threshold = th
    tem_set = []
    for row in data.itertuples():
        udx = row.dx+threshold
        udy = row.dy+threshold
        ddx = row.dx-threshold
        ddy = row.dy-threshold
        tem = data.loc[(data['dx']>=ddx)&(data['dx']<=udx)&(data['dy']>=ddy)&(data['dy']<=udy)]
        if tem.shape[0] < ca_th:
            tem_set.append(row.dx)
    data1=data[~data['dx'].isin(tem_set)]
    data2 = data1[(data1['dx']>=-dx_th)&(data1['dx']<=dx_th)&(data1['dy']>=-dy_th)&(data1['dy']<=dy_th)]
    data3 = data[~((data['dx']>=-dx_th)&(data['dx']<=dx_th)&(data['dy']>=-dy_th)&(data['dy']<=dy_th))]
    new_data = pd.concat([data2,data3],ignore_index=False)
    return new_data

#傅里叶变换滤波平滑曲线
def smooth(data1,Lrk):
    data = data1.copy()
    try:
        data['dx']=data.apply(lambda x:x['minPos']*math.sin(x['Rad']),axis=1)
        data['dy']=data.apply(lambda x:x['minPos']*math.cos(x['Rad']),axis=1)
    except:
        data['dx']=data.apply(lambda x:x['maxPos']*math.sin(x['Rad']),axis=1)
        data['dy']=data.apply(lambda x:x['maxPos']*math.cos(x['Rad']),axis=1)
    x = data['dx'].values
    y = data['dy'].values
    signal = x + 1j*y
    print(str(signal.size))
    # FFT and frequencies
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    # filter
    cutoff = 0.15
    fft[np.abs(freq) > cutoff] = 0
    # IFFT
    filt = np.fft.ifft(fft)
    f_data = pd.DataFrame(columns=['dx'])
    f_data['dx'] = filt.real
    f_data['dy'] = filt.imag
    f_data['Rad'] = f_data.apply(lambda x:math.atan2(x['dx'],x['dy']),axis=1)
    f_data['Pos'] = f_data.apply(lambda x:math.pow((math.pow(x['dx'],2)+math.pow(x['dy'],2)),0.5),axis=1)
    Tail_devourer1 = f_data[0:1]
    f_data = f_data.append(Tail_devourer1,ignore_index=True)
    f_data['Lrk'] = Lrk
    return f_data

def smooth_boun(data1,data2,labels):
    '''
    data1: maxDens
    data2: minDens
    '''
    newMin = []
    newMax = []
    for i in Lrk:
        tem1 = data1[data1['Lrk']==i]
        tem2 = data2[data2['Lrk']==i]
        filt_data1 = smooth(tem1,i)
        filt_data2 = smooth(tem2,i)
        figsize = 4,4
        fig1 = plt.figure(figsize=figsize)
        sns.set_context("paper",font_scale=1.5)
        sns.set(style='ticks')
        ax = fig1.add_subplot(111,projection='polar')
        ax.plot(filt_data1.Rad,filt_data1.Pos,lw=1.5,c='b')
        ax.plot(filt_data2.Rad,filt_data2.Pos,lw=1.5,c='r')
        ax.set_theta_direction('clockwise')#   逆时针
        ax.set_theta_zero_location('N')#   极坐标0°位置为正北
        ax.set_thetagrids(np.arange(0.0,360.0,45.0))#  角度网格线显示
        ax.set_rgrids(np.arange(0.0,10.0,2.0))#    极径网格线显示
        plt.tick_params(labelsize=20)
        plt.savefig(r'C:/Users/dn4/Desktop/论文/smoothBoun'+str(i)+'.png',dpi=600)
        #ax.set_rlabel_position('90')# 极径标签位置
        #ax.set_rlim()#    极径范围设置
        newMin.append(filt_data1)
        newMax.append(filt_data2)
    newMin = pd.concat(newMin,ignore_index=False)
    newMax = pd.concat(newMax,ignore_index=False)
    return newMin,newMax
#%%
def KDE(X,X_plot):
    kde = KernelDensity(kernel='gaussian',bandwidth=0.75 ).fit(X.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    return log_dens

def kde_dis2(data,lim=20):
    dlim= 0
    ulim= lim - 5
    inval= 2000
    X_plot = np.linspace(dlim,ulim,inval)[:,np.newaxis]
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    figsize = 6,5
    fig = plt.figure(figsize=figsize)
    tem1 = data.loc[(data['lenRank']==2)&(data['BRank']==45)]
    X1 = tem1['CSSD'].values
    log_dens1 = KDE(X1,X_plot)
    ax = fig.add_subplot(111)
    ax.plot(X_plot[:,0],np.exp(log_dens1),label="100-200m,45°")
    ax.set_xlabel("CSSD(L)",fontsize=14)
#    ax.set_ylabel("Probability Density",fontsize=14)
    ax.set_ylabel("概率密度",fontsize=14)
    ax.legend(loc='best',fontsize=14)
    plt.tick_params(labelsize=12)
    #
    figsize = 6,5
    fig = plt.figure(figsize=figsize)
    tem2 = data.loc[(data['lenRank']==3)&(data['BRank']==275)]
    X2 = tem2['CSSD'].values
    log_dens2 = KDE(X2,X_plot)
    ax = fig.add_subplot(111)
    ax.plot(X_plot[:,0],np.exp(log_dens2),label="200-300m,275°")
    ax.set_xlabel("CSSD(L)",fontsize=14)
#    ax2.set_ylabel("Probability Density",fontsize=14)
    ax.set_ylabel("概率密度",fontsize=14)
    plt.tick_params(labelsize=12)
    ax.legend(loc='best',fontsize=14)
#%%基本设置
lim1 = 20#用于核密度估计的筛选阈值
lim2 = 10#用于绘图的筛选阈值
#设置图标签
labels = ['50-100m','100-200m','200-300m','300m~']
Lrk = df.lenRank.unique()
#设置核密度估计的带宽
bandwidth = 1
#%%
df1=df_deal(df,lim1)#基本处理
pre_plot()
remark = 'Original'
#rD_dis(df1,remark,labels)#相对位置点分布
#CSSD_dis(df1,remark,labels)#CSSD点分布
minDens = min_cssd(df1)
maxDens=kde_cssd(df1,bandwidth,lim1)
#polar_cssd(maxDens,minDens,labels)
#%%复制处理
dfC = df1.copy()
dfC = dfC.loc[(dfC['dx']>=-lim2) & (dfC['dx']<=lim2) & (dfC['dy']>=-lim2) & (dfC['dy']<=lim2)]
CSSD_dis(dfC,remark,labels)#仅设置上限对df进行筛选，可以视为原CSSD点分布
#a,b=smooth_boun(maxDens,minDens,labels)#a,b无须保存使用
#%%
th1 = 0.15
ca_th1 = 2
dx_th1 = 1.0
dy_th1 = 2.0
data1 = dfC[dfC['lenRank']==1]
new_data1 = cssd_clear(data1,th1,ca_th1,dx_th1,dy_th1)
#%%
th2 = 0.15
ca_th2 = 3
dx_th2 = 1.0
dy_th2 = 2.2
data2 = dfC[dfC['lenRank']==2]
new_data2 = cssd_clear(data2,th2,ca_th2,dx_th2,dy_th2)
#%%
th3 = 0.15
ca_th3 = 2
dx_th3 = 0.8
dy_th3 = 1.5
data3 = dfC[dfC['lenRank']==3]
new_data3 = cssd_clear(data3,th3,ca_th3,dx_th3,dy_th3)
#%%
th4 = 0.08
ca_th4 = 3
dx_th4 = 0.8
dy_th4 = 1.3
data4 = dfC[dfC['lenRank']==4]
new_data4 = cssd_clear(data4,th4,ca_th4,dx_th4,dy_th4)
#%%
dfD = pd.concat([new_data1,new_data2,new_data3,new_data4],ignore_index=False)
remark = 'Processed'
CSSD_dis(dfD,remark,labels)#CSSD点分布
#%%
minDens_D = min_cssd(dfD)
maxDens_D=kde_cssd(dfD,bandwidth,lim1)
polar_cssd(maxDens_D,minDens_D,labels)
Max,Min = smooth_boun(maxDens_D,minDens_D,labels)
kde_dis2(dfD)
#%%输出结果
#maxDens_D.to_csv('D:/结果/boundary/maxDens.csv',index=0,header=True)
#minDens_D.to_csv('D:/结果/boundary/minDens.csv',index=0,header=True)
#Min.to_csv(outputFile1,index=0,header=True)
#Max.to_csv(outputFile2,index=0,header=True)