# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:50:42 2020

@author: 24592
"""

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans
from sklearn import metrics
#输入文件
#‪D:\结果\01confResult.csv
inputFile01 = 'D:/结果/01confResult.csv'
inputFile02 = 'D:/结果/02confResult.csv'
inputFile03 = 'D:/结果/03confResult.csv'
inputFile06 = 'D:/结果/06confResult.csv'
inputFile07 = 'D:/结果/07confResult.csv'
inputFile16 = 'D:/结果/16confResult.csv'
inputFile17 = 'D:/结果/17confResult.csv'
inputFile18 = 'D:/结果/18confResult.csv'
inputFile19 = 'D:/结果/19confResult.csv'
inputFile20 = 'D:/结果/20confResult.csv'
inputFile21 = 'D:/结果/21confResult.csv'
inputFile22 = 'D:/结果/22confResult.csv'
inputFile23 = 'D:/结果/23confResult.csv'
inputFile24 = 'D:/结果/24confResult.csv'
df_01 = pd.read_csv(inputFile01,encoding='gbk',engine='python')
df_02 = pd.read_csv(inputFile02,encoding='gbk',engine='python')
df_03 = pd.read_csv(inputFile03,encoding='gbk',engine='python')
df_06 = pd.read_csv(inputFile06,encoding='gbk',engine='python')
df_07 = pd.read_csv(inputFile07,encoding='gbk',engine='python')
df_16 = pd.read_csv(inputFile16,encoding='gbk',engine='python')
df_17 = pd.read_csv(inputFile17,encoding='gbk',engine='python')
df_18 = pd.read_csv(inputFile18,encoding='gbk',engine='python')
df_19 = pd.read_csv(inputFile19,encoding='gbk',engine='python')
df_20 = pd.read_csv(inputFile20,encoding='gbk',engine='python')
df_21 = pd.read_csv(inputFile21,encoding='gbk',engine='python')
df_22 = pd.read_csv(inputFile22,encoding='gbk',engine='python')
df_23 = pd.read_csv(inputFile23,encoding='gbk',engine='python')
df_24 = pd.read_csv(inputFile24,encoding='gbk',engine='python')
df = pd.concat([df_01,df_02,df_03,df_06,df_07,df_16,df_17,df_18,df_19,df_20,df_21,df_22,df_23,df_24],ignore_index=True)
df['duration'] = df.apply(lambda x:x['Duration']/60,axis=1)
df=df.loc[(df.meanSev<1.2)&(df.maxSev<1.5)&(df.duration<7)]
#提取每次冲突所在的时间段（小时）
df['Hour'] = df.apply(lambda x:time.strftime("%H",time.localtime(x['time'])),axis=1)
#%%
x = df[['meanSev','duration','maxSev']].values
x_scaled = preprocessing.scale(x)
#%%

figsize = 11,14
fig = plt.figure(figsize=figsize)
fig.suptitle('Standardization',fontsize=18)
ax=plt.subplot(2,3,1)
ax.set_title('meanSev')
ax.hist(x[:,0])
ax=plt.subplot(2,3,2)
ax.set_title('duration')
ax.hist(x[:,1])
ax=plt.subplot(2,3,3)
ax.set_title('maxSev')
ax.hist(x[:,2])
ax=plt.subplot(2,3,4)
ax.set_title('meanSev(processed)')
ax.hist(x_scaled[:,0])
ax=plt.subplot(2,3,5)
ax.set_title('duration(processed)')
ax.hist(x_scaled[:,1])
ax=plt.subplot(2,3,6)
ax.set_title('maxSev(processed)')
ax.hist(x_scaled[:,2])
#%%
#figsize = 11,14
#fig = plt.figure(figsize=figsize)
#ax = Axes3D(fig)
#ax.scatter(x[:,0],x[:,1],x[:,2])
#%%聚类，轮廓值法
score = []
for i in range(2,11,1):   
    kmeans = KMeans(n_clusters=i,random_state=1).fit(x_scaled)
    X_Scaled = pd.DataFrame(x_scaled)
    labels = kmeans.labels_
    #轮廓值法
    score.append(metrics.silhouette_score(x_scaled,labels, metric='euclidean'))   
#    score2.append(metrics.calinski_harabasz_score(x_scaled,labels))       
maxIndex = score.index(max(score))+2
print('轮廓值法的最佳聚类中心数量：'+ str(maxIndex))
figsize = 11,14
fig = plt.figure(figsize=figsize)
fig.suptitle('Silhouette Coefficient',fontsize=18)
plt.plot(range(2,11,1),score)
#%%根据最佳的聚类数重新聚类
kmeans = KMeans(n_clusters=maxIndex,random_state=1).fit(x_scaled)
X_Scaled = pd.DataFrame(x_scaled)
labels = kmeans.labels_
X_Scaled['labels'] = labels
df['labels'] = labels
lists = df['labels'].unique()
figsize = 11,14
fig = plt.figure(figsize=figsize)
fig.suptitle('n_chusters: '+str(i),fontsize=18)
ax = Axes3D(fig)
for j in lists:
    tem_df = X_Scaled[X_Scaled['labels']==j]
    ax.scatter(tem_df[0],tem_df[1],tem_df[2])
    ax.set_xlabel('meanSev', fontdict={'size': 15, 'color': 'k'})
    ax.set_ylabel('duration', fontdict={'size': 15, 'color': 'k'})
    ax.set_zlabel('maxSev', fontdict={'size': 15, 'color': 'k'})
#%%不同模式的平均生命周期


#figsize = 11,14
#fig = plt.figure(figsize=figsize)
##fig.suptitle('Standardization',fontsize=18)
#plt.plot(range(df.shape[0]),df.meanSev)