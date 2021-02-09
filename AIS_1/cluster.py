# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:22:11 2020

@author: 24592
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_read(path):
    '''
    文件读取
    '''
    df = []
    for root,dirs,files in os.walk(path):
        for f in files:
            data = pd.read_csv(os.path.join(root,f))
            df.append(data)
    df = pd.concat(df,ignore_index=False)
    df.drop('count',axis=1,inplace=True)
    return df

def rd_deC(cc):
    '''
    画rd与deCon的相对关系图
    '''
    t = cc.timestamp.tolist()
    x = [a-t[0] for a in t]
#    a = np.trapz(cc.deCon.tolist(),x)/(max(x)-min(x))
    y1 = cc.deCon.tolist()
    y2 = cc.RD.tolist()
    data = {'deCon':y1,'RD':y2}
    tem = pd.DataFrame(data, index=x)
    ax = tem.plot(secondary_y=['RD'],x_compat=True,grid=True)
    ax.set_title("RD-deCon")
    ax.set_ylabel('deCon')
    ax.grid(linestyle="--", alpha=0.3)
    ax.right_ax.set_ylabel('RD')
    plt.show()
    
#    plt.figure()
#    y3 = cc.oSog.tolist()
#    y4 = cc.tSog.tolist()
#    plt.plot(x,y3)
#    plt.plot(x,y4)
    
#    plt.figure()
#    y5 = cc.RaApp.tolist()
#    plt.plot(x,y5)
#    plt.show()

def con_filter(df,timeshrold,display=False):
    '''
    过滤冲突数据
    '''
    newdf = []
    for group in df.groupby('ID'):
        time = group[1]['timestamp'].tolist()
        new = [time[i] -time[i-1] for i in range(len(time))]
        new.pop(0)
        new.insert(0,0)
        index = []
        for i,t in enumerate(new):
            if i < 1 or t > timeshrold or i == len(new)-1:
                c = i
                index.append(c)
        if len(index) > 2:
            for j,idx in enumerate(index):
                if j > 1:
                    data = group[1][index[j-1]:idx]
                    deCon = data['deCon'].tolist()
                    if deCon[0] < 0.05 and deCon[-1] < 0.05:
                        ddd = data
                        newdf.append(ddd)
        else:
            deCon = group[1]['deCon'].tolist()
            if deCon[0] < 0.05 and deCon[-1] < 0.05:
                ddd = group[1]
                newdf.append(ddd)
        if display != False:
            if len(newdf) == display:
                ccc = ddd
                rd_deC(ccc)
    return newdf,len(newdf)

def area_cal(list1,list2):
    '''
        list2:时间
        求list1中点与时间轴围城的面积
        及比例
    '''
    area = np.trapz(list1,list2)
    t = list2[-1] - list2[0]
    at = area/t
    return area,at

def cal(newdf):
    result = []
    for data in newdf:
        oM = data['oM'].values[0]
        tM = data['tM'].values[0]
        Lrk = data['Lrk'].values[0]
        maxDeg = data.deCon.max()
        deCon = data.deCon.tolist()
        time = data.timestamp.tolist()
        i = deCon.index(maxDeg)
        x = [a - time[0] for a in time]
        t1 = time[0]
        t2 = time[i]
        t3 = time[-1]
        duration = x[-1] - x[0]
        _,meanDeg = area_cal(deCon,x)
        D1 = deCon[0:i+1]
        x1 = x[0:i+1]
        D2 = deCon[i:]
        x2 = x[i:]
        if len(x1) > 1 and len(x2) > 1:
            _,R1 = area_cal(D1,x1)
            _,R2 = area_cal(D2,x2)
            r1 = (D1[-1]-D1[0])/(x1[-1]-x1[0])
            r2 = (D2[0]-D2[-1])/(x2[-1]-x2[0])
            P1 = R1/maxDeg
            P2 = R2/maxDeg#计算公式area/(maxDeg*T)
            P = (P1+P2)/2
            res = [oM,tM,t1,t2,t3,Lrk,maxDeg,meanDeg,duration,r1,r2,P]
            result.append(res)
    result = pd.DataFrame(result,columns=['oM','tM','t1','t2','t3','Lrk','maxDeg','meanDeg','duration','r1','r2','P'])
    Result = result.loc[(result.r2<0.03)]
    return Result

def minmax(X):
    '''
        01标准化
    '''
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    return X_minmax

def km(data,k):
    '''
        kmeans
    '''
    from sklearn.cluster import KMeans
    from sklearn import metrics
    km = KMeans(n_clusters=k,init='k-means++',random_state=1).fit(data)
    labels = km.labels_
    #center = km.cluster_centers_
    score = metrics.silhouette_score(data, labels, metric='euclidean')
    return labels,score

def pca(data):
    '''
        pca降维为2维
    '''
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(data)
    #pca.explained_variance_ratio_
    #pca.singular_values_
    return result

def pro_da(data,attribute):
    X = data[attribute]
    X_minmax = minmax(X)
    X_pca = pca(X_minmax)    
    return X_pca

def score_dis(K,Score):
    plt.figure()
    eee = pd.DataFrame({'K':K,'Score':Score})
    sns.lineplot(x='K',y='Score',data = eee,
                 markers='*',legend='full')
    
def d3_plot(data):
    '''
        3d数据展示
    '''
    from mpl_toolkits.mplot3d import Axes3D
    figsize = 11,14
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax = Axes3D(fig)
    c_kinds = ('r','b','m','y','g')
    if data.shape[1] == 3:
        ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2])
    else:
        for group in data.groupby('label'):
            tem = group[1]
            i = group[0]
            ax.scatter(tem.iloc[:,0],tem.iloc[:,1],tem.iloc[:,2],\
                       c=c_kinds[int(i)])
    ax.set_xlabel(data.columns[0], fontdict={'size': 15, 'color': 'k'})
    ax.set_ylabel(data.columns[1], fontdict={'size': 15, 'color': 'k'})
    ax.set_zlabel(data.columns[2], fontdict={'size': 15, 'color': 'k'})
    
def d2_plot(data):
    '''
        2d数据展示
    '''
    figsize = 11,14
    plt.figure(figsize=figsize)
    ax = plt.gca()
    c_kinds = ('r','b','m','y','g')
    if data.shape[1] == 2:
        ax.scatter(data.iloc[:,0],data.iloc[:,1])
    else:
        for group in data.groupby('label'):
            tem = group[1]
            i = group[0]
            ax.scatter(tem.iloc[:,0],tem.iloc[:,1],c=c_kinds[int(i)])
    
def best_k(X_pca):
    print('Kmeans begin')
    Score = []
    K = [a for a in range(2,11,1)]
    for k in range(2,11,1):
        _,score = km(X_pca,k)
        Score.append(score)
    score_dis(K,Score)
    best_k = K[Score.index(max(Score))]
    labels,_ = km(X_pca,best_k)
    #xx = X.copy()
    #xx['label'] = labels
    #d3_plot(xx)
#    X_pca = pd.DataFrame(X_pca)
#    xxx = X_pca.copy()
#    xxx['label'] = labels
#    d2_plot(xxx)
    return labels

def gmm(X_pca,data,attribute):
    print('GMM begin')
    from sklearn import mixture
    n_components = np.arange(1,21)
    for i in range(50):
        gmmM = [mixture.GaussianMixture(n,covariance_type='full',max_iter=100).fit(X_pca) for n in n_components]
        bic = [m.bic(X_pca) for m in gmmM]
        print(str(i)+": "+str(bic.index(min(bic))))
        if bic.index(min(bic)) == 3:
            print('find you!!!')
            labels = gmmM[bic.index(min(bic))].predict(X_pca)
            plt.figure()
            plt.plot(n_components,bic,label='BIC')
            plt.legend(loc='best')
            plt.xlabel('n_components')
            plt.show()
            X1 = data[attribute].copy()
            X1['label'] = labels
            X1 = label_reset(X1)
            if len(attribute) == 3:
                d3_plot(X1)
            else:
                x2 = pd.DataFrame(X_pca)
                x2['label'] = labels
                d2_plot(x2)
            break
    return labels

def box_plot(data1,attribute):
    plt.figure(figsize=(10,10))
    sns.set(style='whitegrid')
#    plt.subplots_adjust(bottom=.04, top=0.95, hspace=.2, wspace=.05,left=.03, right=.97)
    for i in attribute:
        if len(attribute)==4:
            plt.subplot(2,2,attribute.index(i)+1)
        else:
            plt.subplot(1,len(attribute),attribute.index(i)+1)
        sns.boxplot(x='label',y=i,data=data1)
        plt.show()

def label_reset(data):
    '''
        重设标签
    '''
    data1 = data.copy()
    label = data1['label'].unique()
    label.sort()
    M = []
    for l in label:
        tem = data1[data1['label'].isin([l])]
        m = tem['maxDeg'].mean()
        M.append(m)
    M = pd.DataFrame(M,columns=['value'])
    M = M.sort_values(by='value')
    L = M.index.tolist()
    tem_L = ['s1','s2','s3','s4','s5','s6']
    tem_LL = [1,2,3,4,5,6]
    tem_l = tem_L[0:len(L)]
    tem_ll = tem_LL[0:len(L)]
    data1['label'].replace(L,tem_l,inplace=True)
    data1['label'].replace(tem_l,tem_ll,inplace=True)
    return data1

def alg_choose(data,X_pca,attribute,n_style=0):
    if n_style == 1:
        labels = best_k(X_pca)
    else:
        labels = gmm(X_pca,data,attribute)
    data1 = data.copy()
    data1['label'] = labels
    data2 = label_reset(data1)
    box_plot(data2,attribute)
    return data2

def main():
    df = data_read("D:/0924result/")
    timeshrold = 300
    display = False#输入数字可画图
    newdf,count = con_filter(df,timeshrold,display)
    result = cal(newdf)
#    data = result.loc[(result['maxDeg']<=1)&(result['duration']<=1000)]#排除存在碰撞危险的数据
    attribute = ['Severity','duration']#选取属性
    X_pca = pro_da(data,attribute)
    try:
        Result = alg_choose(data,X_pca,attribute,n_style=0)#默认选取GMM
    
        Result.to_csv('D:/结果/result_0406.csv',sep=',',header=True,index=0)
    except UnboundLocalError:
        print('Search failed!!!') 
    
if __name__ == '__main__':
    main()