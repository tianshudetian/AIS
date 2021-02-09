# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:40:39 2020

@author: dn4
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def minmax(X):
    '''
        01标准化
    '''
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    return X_minmax

def gmm(X):
    from sklearn import mixture
    n_components = np.arange(1,21)
    gmmM = [mixture.GaussianMixture(n,covariance_type='full',max_iter=100).fit(X) for n in n_components]
    bic = [m.bic(X) for m in gmmM]
    labels = gmmM[bic.index(min(bic))].predict(X)
    return n_components,bic,labels
#%%
'''
    三个输出：
    (1)网络规模N
    (2)边数量E
    (3)平均入/出接近中心性Cin/Cout
'''
def twoShip(data):
    '''
        两船冲突属性数据计算与输出
    '''
    tem = data.copy()
    N = 2#两船冲突规模为2
    sev1 = tem.sev1.values[0]
    sev2 = tem.sev2.values[0]
    if sev1 == 0 or sev2 == 0:
        E = 1
        if sev1 == 0:
            Cin = sev2
            Cout = Cin
        else:
            Cin = sev1
            Cout = Cin
    else:
        E = 2
        Cin = (sev1+sev2)/2
        Cout = Cin
    res = [N,E,Cin,Cout]
    return res

def multiShip(data):
    '''
        多船冲突属性数据计算与输出
    '''
    tem2 = data.copy()
    Node_state = []
    a = list(tem2['sev1'])
    b = list(tem2['sev2'])
    a.extend(b)
    E = len(shan0(a))
    N = len(items)
    for item in items:
        temp1 = tem2[tem2['mmsi1'].isin([item])]
        temp2 = tem2[tem2['mmsi2'].isin([item])]
        if temp1.shape[0] > 0:
            Sin1 = temp1['sev1'].sum()#当前船舶节点的感知到的冲突程度
            Sout1 = temp1['sev2'].sum()#当前船舶节点的输出的冲突程度
        else:
            Sin1 = 0
            Sout1 = 0
        if temp2.shape[0] > 0:
            Sin2 = temp2['sev2'].sum()#当前船舶节点的感知到的冲突程度
            Sout2 = temp2['sev1'].sum()#当前船舶节点的输出的冲突程度
        else:
            Sin2 = 0
            Sout2 = 0
        Sin = (Sin1+Sin2)/(N-1)
        Sout = (Sout1+Sout2)/(N-1)
        state = [item,Sin,Sout]
        Node_state.append(state)
    Cin = CinCout(Node_state,1)
    Cout = CinCout(Node_state,2)
    res = [N,E,Cin,Cout]   
    return res

def quchong(data):
    '''
        去除重复以及被包含的组
    '''
    temx = data.copy()
    try:
        for i in range(len(temx)):
            p = set(temx[i])
            for j in range(len(temx)):
                if i != j:
                    q = set(temx[j])
                    if p <= q:
                        temx.remove(temx[j])
                        return temx,len(temx)
    except:
        none = 1
        return none
def fenzu(data):
    '''
        对船舶进行匹配分组
    '''
    tem  = data.copy()
    A = []
    for row in tem.itertuples():
        tem = [row.mmsi1,row.mmsi2]
        A.append(tem)
    
    temx = []
    for k in range(len(A)):
        data = A[k]
        for i in range(len(A)):
            if i != k:
                for item in A[i]:
                   if item in data: 
                       data.extend(A[i])#合并list
                       data = list(set(data))#list去重
        temx.append(data)
    
    for k in range(20):
        a = len(temx)
        try:
            temx,b = quchong(temx)
            if a == b:
                break
        except:
            continue
    return temx

def conf_group(list1,data):
    '''
        输出多船冲突场景中的船舶组
    '''
    items = list1.copy()
    tem1 = data.copy()
    Temp = []
    for item in items:
        temp1 = tem1[tem1['mmsi1'].isin([item])]
        temp2 = tem1[tem1['mmsi2'].isin([item])]
        if temp1.shape[0] > 0:
            Temp.append(temp1)
        if temp2.shape[0] > 0:
            Temp.append(temp2)
    Temp = pd.concat(Temp)   
    Temp = Temp.drop_duplicates()
    return Temp

def CinCout(data,para=1):
    '''
        para=1表示求Cin
        para=2表示求Cout
    '''
    Node_state = data.copy()
    S_sum = 0
    num = 0
    for i in Node_state:
        if i[para] != 0:
            S_sum = S_sum + i[para]
            num = num + 1
    C = S_sum/num
    return C

def shan0(a):
    '''
        删除list中的0元素
    '''
    c = []
    for i in a:
        if i != 0:
            c.append(i)
    return c
#%%
df = []
for info in os.listdir(r'D:/11.24结果/result'): 
    domain = os.path.abspath(r'D:/11.24结果/result') #获取文件夹的路径 
    info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径 
    data = pd.read_csv(info,encoding='gbk',engine='python') 
    df.append(data)
df = pd.concat(df)
df = df.loc[(df['sev1']<=4.075)&(df['sev2']<=4.075)&(df['sev1']>=0)&(df['sev2']>=0)]
Times = df.time.unique()

T = []
T.append(Times[0])
for i in range(len(Times)):
    try:
        diff = Times[i+1] - Times[i]
        if diff > 10:
            T.append(Times[i+1])
    except:
        continue
#%%
Res = []
for t in T:
    for j in range(0,10000,10):
        tem = df[df['time'].isin([t+j])]
        if tem.shape[0] > 0:
            if tem.shape[0] == 1:
                res = twoShip(tem)
            elif tem.shape[0] > 1:
                tem1 = tem.copy()
                temx = fenzu(tem)
                for items in temx:
                    tem2 = conf_group(items,tem1)#多船冲突船舶组
                    if tem2.shape[0] == 1:
                        res = twoShip(tem2)
                    else:
                        res = multiShip(tem2)
            Res.append(res)
        else:
            break
Res = pd.DataFrame(Res)
Res.set_axis(['N','E','Cin','Cout'],axis='columns',inplace=True)
#%%
temn = Res.iloc[:,:3]
em = minmax(temn)
n,b,l=gmm(em)
plt.plot(b)
