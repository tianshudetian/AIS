# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:23:17 2020

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

def gmm(X):
    from sklearn import mixture
    n_components = np.arange(1,21)
    gmmM = [mixture.GaussianMixture(n,covariance_type='full',max_iter=100).fit(X) for n in n_components]
    bic = [m.bic(X) for m in gmmM]
    labels = gmmM[bic.index(min(bic))].predict(X)
    return n_components,bic,labels

def kmeans(X):
    from sklearn.cluster import KMeans
    n_clusters = np.arange(1,21)
    kmeans = [KMeans(n, random_state=0).fit(X) for n in n_clusters]
    score = [m.score(X) for m in kmeans]
    return score
#
#df = []
#for info in os.listdir(r'D:/11.24结果/res'): 
#    domain = os.path.abspath(r'D:/11.24结果/res') #获取文件夹的路径 
#    info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径 
#    data = pd.read_csv(info,encoding='gbk',engine='python') 
#    df.append(data)
#df = pd.concat(df)
#
#df['duration'] = df.apply(lambda x:(x['T3']-x['T1'])/60,axis=1)
#df = df.loc[(df['R1']<=0.0267)&(df['R2']>=-0.0254)&(df['maxSev']<=3.33)&(df['duration']<=4.5)]
#temp = df.copy()
#em = minmax(df.iloc[:,1:4])

#df.to_csv(r'D:/11.24结果/data.csv')
df = pd.read_csv(r'D:/11.24结果/data.csv',encoding='gbk',engine='python')
temp = df.copy()
em = minmax(df.iloc[:,2:5])
#emm = pca(em)
#n,b,l=gmm(emm)
#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
SSE = []
Scores = []  # 存放轮廓系数
for k in range(2,21):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(em)
    Scores.append(silhouette_score(em,estimator.labels_,metric='euclidean'))
    SSE.append(estimator.inertia_)
X = range(2,21)
#%%
xticks = range(2,21,2)
figsize = 6,4
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
plt.plot(X,Scores,
         linestyle=":", #线形
         linewidth=2, #线粗
         color = '#0000FF', #线的颜色
         marker = '*', #点的形状
         markersize = 7, #点的大小
         markeredgecolor='#FF0000', # 点的边框色
         markerfacecolor='#FF0000')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

ax.set_xlabel('The number of clusters',fontsize=17,fontname = 'Times New Roman')
ax.set_ylabel('Silhouette Coefficient',fontsize=17,fontname = 'Times New Roman') 
plt.xticks(xticks)
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()
plt.savefig(r'C:/Users/dn4/Desktop/论文/SC.png',dpi=600)

figsize = 6,4
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

plt.plot(X,SSE,
         linestyle=":", #线形
         linewidth=2, #线粗
         color = '#0000FF', #线的颜色
         marker = '*', #点的形状
         markersize = 7, #点的大小
         markeredgecolor='#FF0000', # 点的边框色
         markerfacecolor='#FF0000')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

ax.set_xlabel('The number of clusters',fontsize=17,fontname = 'Times New Roman')
ax.set_ylabel('Sum of the Squared Errors (SSE)',fontsize=17,fontname = 'Times New Roman') 
plt.xticks(xticks)
plt.tick_params(labelsize=16)
plt.grid(linestyle='--')
plt.show()
plt.savefig(r'C:/Users/dn4/Desktop/论文/SSE.png',dpi=600)#C:\Users\dn4\Desktop\论文
#%%
silhouette_all=[]
from sklearn import mixture
from sklearn import metrics
for k in range(2,21):
    gmm_model = mixture.GaussianMixture(k,covariance_type='full',max_iter=100).fit(em)
    labels = gmm_model.predict(em)
    a = metrics.silhouette_score(em, labels, metric='euclidean')
    silhouette_all.append(a)
plt.plot(X,silhouette_all)
#%%
#from sklearn.cluster import DBSCAN
#clustering = DBSCAN(eps=0.1, min_samples=2).fit(em)
#clustering.labels_
#
#clustering
##%%
#
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

x = df.iloc[:, 1]  # [ 0  3  6  9 12 15 18 21]
y = df.iloc[:, 2]  # [ 1  4  7 10 13 16 19 22]
z = df.iloc[:, 3]  # [ 2  5  8 11 14 17 20 23]
 
 
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
 
 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('R2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('R1', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('maxSev', fontdict={'size': 15, 'color': 'red'})
plt.show()
#%%
#estimator = KMeans(n_clusters=4)  # 构造聚类器
#estimator.fit(em)
#labels = estimator.predict(em)
#temp['label'] = labels
#temp.to_csv(r'D:/11.24结果/clusterReslut.csv')
temp = pd.read_csv(r'D:/11.24结果/clusterReslut.csv',encoding='gbk',engine='python')
Label = ['C1','C2','C3','C4']

for i in range(0,3,1):
    all_list = []
    for j in range(0,4,1):
        tem = temp[temp['label'].isin([j])]
        List = list(tem.iloc[:,i+2])
#        List = list(tem.iloc[:,i+1])
#        print(str(j+1)+': '+str(len(List)))
        if i == 0:
            percentile = np.percentile(List, (25, 50, 75), interpolation='linear')
            Q1 = percentile[0]#上四分位数
            Q2 = percentile[1]
            Q3 = percentile[2]#下四分位数
            IQR = Q3 - Q1#四分位距
            ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
            llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
            num = len(List)
            print(str(num)+' '+str(0.01*num/(Q3-Q1))+' '+str(0.01*num/(ulim-llim)))
        all_list.append(List)
    figsize = 5,4
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    box = plt.boxplot(all_list,
                patch_artist=True,#7B68EE
#                boxprops=dict(color="blue"),
                showmeans=True,
                flierprops = {'marker':'o','markerfacecolor':'red','markeredgecolor':'black','markersize':3},
                meanprops = {'marker':'D','markerfacecolor':'#FFA500','markeredgecolor':'#FAF0E6','markersize':4},
                medianprops = {'linestyle':'--','color':'#D8BFD8'}
                               )
    colors = ['#6495ED', '#20B2AA', '#FF8C00', '#9370DB', '#7B68EE']
    for patch, color in zip(box['boxes'], colors):
       patch.set_facecolor(color)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]    
    
    plt.xticks(range(1,5,1), Label, rotation=0)  
    #ax.set_xlabel('Cluster',fontsize=17,fontname = 'Times New Roman')
    
#    plt.rcParams["text.usetex"]
#    plt.rcParams["mathtext.fontset"]
    if i == 0:
        ax.set_ylabel(r'$\mathrm{S_{max}}$',fontsize=17)
        name = 'Smax.png'
#        ax.set_ylabel(r'$ S_{max} $',fontsize=17) 
    elif i == 1:
        ax.set_ylabel(r'$\mathrm{R}_1$',fontsize=17,fontname = 'Times New Roman')
        name = 'R1.png'
    else:
        ax.set_ylabel(r'$\mathrm{R}_2$',fontsize=17,fontname = 'Times New Roman') 
        name = 'R2.png'
    #ax.set_ylabel(r'$\mathrm{S}_{\mathrm{max}}$',fontsize=17) 
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')
#    plt.show()
    plt.savefig(r'C:/Users/dn4/Desktop/论文/'+name,dpi=600)
#%%
import time
t = temp['time'].apply(lambda x:time.strftime("%H", time.localtime(x)))
temp['t']=t.astype('int32')
C1 = []
C2 = []
C3 = []
C4 = []

for i in range(0,24,1):
    tem = temp[temp['t'].isin([i])]
    for j in range(0,4,1):
        tem1 = tem[tem['label'].isin([j])]
        if j == 0:
            C1.append(len(tem1))
        elif j == 1:
            C2.append(len(tem1))
        elif j == 2:
            C3.append(len(tem1))
        else:
            C4.append(len(tem1))
#%%
figsize = 9,3.5
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
x = range(0,24,1)
plt.bar(x,C1,color='#87CEFA',label='C1')
plt.bar(x,C2,color='#00BFFF',bottom = C1,label='C2')
plt.bar(x,C3,color='#1E90FF',bottom = np.sum([C1,C2],axis=0),label='C3')
plt.bar(x,C4,color='#0000CD',bottom = np.sum([C1,C2,C3],axis=0),label='C4')
ax.set_xlabel('Time [hour]',fontsize=15,fontname = 'Times New Roman')
ax.set_ylabel('The number of conflict',fontsize=15,fontname = 'Times New Roman')
plt.legend(loc="upper right")
xticks = range(0,24,1)
plt.xticks(xticks)
plt.tick_params(labelsize=14)
plt.grid(linestyle='--')
plt.show()
