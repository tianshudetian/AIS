# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:39:13 2020

@author: dn4
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
inputFile = r'D:/11.24结果/clusterReslut.csv'
df = pd.read_csv(inputFile,encoding='gbk',engine='python')
temp = df.copy()
Label = ['C1','C2','C3','C4']
#%%
for i in range(0,3,1):
    all_list = []
    for j in range(0,4,1):
        tem = temp[temp['label'].isin([j])]
        List = list(tem.iloc[:,i+2])
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
#        ax.set_ylabel(r'$ S_{max} $',fontsize=17) 
    elif i == 1:
        ax.set_ylabel(r'$\mathrm{R}_1$',fontsize=17,fontname = 'Times New Roman')
    else:
        ax.set_ylabel(r'$\mathrm{R}_2$',fontsize=17,fontname = 'Times New Roman') 
    #ax.set_ylabel(r'$\mathrm{S}_{\mathrm{max}}$',fontsize=17) 
    plt.tick_params(labelsize=16)
    plt.grid(linestyle='--')
    plt.show()
#%%
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
figsize = 9,3.5
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]   
x = range(0,24,1)
ln1 = plt.bar(x,C1,color='#87CEFA',label='C1')
ln2 = plt.bar(x,C2,color='#00BFFF',bottom = C1,label='C2')
ln3 = plt.bar(x,C3,color='#1E90FF',bottom = np.sum([C1,C2],axis=0),label='C3')
ln4 = plt.bar(x,C4,color='#0000CD',bottom = np.sum([C1,C2,C3],axis=0),label='C4')
ax.set_xlabel('Local time [hour]',fontsize=15,fontname = 'Times New Roman')
ax.set_ylabel('The number of conflict',fontsize=15,fontname = 'Times New Roman')
plt.legend(loc="upper left",prop={'family': 'Times New Roman','size':14})
ax1 = ax.twinx()
count = np.load(r'D:/论文数据/ShipCount.npy')
c = [(i + 20)/10 for i in count]
ax1.plot(x,c)
xticks = range(0,24,2)
plt.xticks(xticks)
plt.tick_params(labelsize=14)
#lns = ln1 + ln2 + ln3 + ln4
#labs = [ln.get_label() for ln in lns]
#font1 = {'family' : 'Times New Roman',
#         'weight' : 'normal',
#         'size'   : 14,
#         }
#ax.legend(lns,labs,loc='lower left',prop=font1,ncol=2, bbox_to_anchor=(0,1.02,1,0.2),mode='expand')
plt.grid(linestyle='--')
plt.show()
#%%
temp['duration'] = temp.apply(lambda x:(x['T3']-x['T1'])/60,axis=1)
temp1=temp.loc[(temp['duration']<=48.7)]
figsize = 5,4
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
for i in range(0,4,1):
    tem = temp1[temp1['label'].isin([i])]
    if i == 0:
        tem = tem.loc[(tem['duration']<=16.314)]
    elif i == 1:
        tem = tem.loc[(tem['duration']<=9.862)]
    print(str(i)+': '+str(tem.duration.max()))
    tem.duration.plot(kind='kde',label=Label[i])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.set_xlabel('Duration [minute]',fontsize=15,fontname = 'Times New Roman')
ax.set_ylabel('Probability density',fontsize=15,fontname = 'Times New Roman')
plt.xlim(-2,46)
plt.legend(loc="upper right",prop={'family': 'Times New Roman','size':14})
plt.tick_params(labelsize=14)
plt.grid(linestyle='--')
plt.show()
plt.savefig(r'C:/Users/dn4/Desktop/论文/duration.png',dpi=600)
