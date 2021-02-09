# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:48:29 2020

@author: dn4
"""

import os 
import pandas as pd
import numpy as np



def read(fileName):
    df = []
    for info in os.listdir(fileName): 
        domain = os.path.abspath(fileName) #获取文件夹的路径
        info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
        data = pd.read_csv(info,encoding='gbk',engine='python') 
        df.append(data)
    df = pd.concat(df)
    return df

def timeTransform(df):
    data = df.copy()
    import time
    T = data.apply(lambda x:time.strftime("%H", time.localtime(x['timestamp'])),axis=1)
    data['T'] = T.astype('int32')
    return data

def timeCount(df):
    data = df.copy()
    count = []
    for i in range(0,24,1):
        temp = data[data['T'].isin([i])][['T','MMSI']]
        temp.drop_duplicates(subset='MMSI',keep='first',inplace=True)
        count.append(temp.shape[0])
    return count
#%%
def countPlot(count):
    count = [i/10 for i in count]
    import matplotlib.pyplot as plt
    figsize = 9,3.5
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    x = range(0,24,1)
#    plt.bar(x,count,color='#87CEFA')
    plt.plot(x,count,color='#87CEFA')
    ax.set_xlabel('Local time [hour]',fontsize=17,fontname = 'Times New Roman')
    ax.set_ylabel('Traffic volume',fontsize=17,fontname = 'Times New Roman')
    xticks = range(0,24,2)
    plt.xticks(xticks)
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')
    plt.show()
#%%
#D:\论文数据\processed2s
#df = read(r'D:/论文数据/dynamicNofilter')
df1 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181001.csv',encoding='gbk',engine='python')
df2 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181016.csv',encoding='gbk',engine='python')
df3 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181017.csv',encoding='gbk',engine='python')
df4 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181018.csv',encoding='gbk',engine='python')
df5 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181019.csv',encoding='gbk',engine='python')
df6 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181020.csv',encoding='gbk',engine='python')
df7 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181021.csv',encoding='gbk',engine='python')
df8 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181022.csv',encoding='gbk',engine='python')
df9 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181023.csv',encoding='gbk',engine='python')
df10 = pd.read_csv(r'D:/论文数据/dynamicNofilter/20181024.csv',encoding='gbk',engine='python')
df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
df1 = timeTransform(df)
count = timeCount(df1)
countPlot(count)
np.save(r'D:/论文数据/ShipCount.npy',count)
