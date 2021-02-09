# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:18:59 2020

@author: dn4
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver
from math import cos,sin,radians

df1 = pd.read_csv("C:/Users/dn4/Desktop/multiVessel.csv")
df2 = pd.read_csv("C:/Users/dn4/Desktop/multiVesselInformation.csv")
txt = [r"$v_1$",r"$v_2$",r"$v_3$",r"$v_4$",r"$v_5$"]
mmsi = df2.mmsi

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
        if row.Index == 0 or row.Index == 1:
            quiver(row.lon,row.lat,dx/2,dy/2, angles='xy', scale=1.0, scale_units='xy', width=0.008)
        else:
            quiver(row.lon,row.lat,dx,dy, angles='xy', scale=1.0, scale_units='xy', width=0.008)
#        ax2.text(row.lon-0.001,row.lat-0.002,txt[row.Index],fontsize=17)
        if row.Index == 0:
            ax2.text(row.lon-0.0003,row.lat-0.0005,txt[row.Index],fontsize=17)
        elif row.Index == 3:
            ax2.text(row.lon-0.0003,row.lat+0.0003,txt[row.Index],fontsize=17)
        else:
            ax2.text(row.lon-0.0002,row.lat-0.0005,txt[row.Index],fontsize=17)
        
    ax2.set_xlim(122.13,122.16)
    ax2.set_ylim(29.913,29.92)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax2.get_xaxis().get_major_formatter().set_scientific(False)
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False    
    
    ax2.set_xlabel('经度(°)',fontsize=17)
    ax2.set_ylabel('纬度(°)',fontsize=17) 
    plt.tick_params(labelsize=17)
    plt.show()

figsize = 6,5
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
#ax.scatter(df2['lon'],df2['lat'],c='black')
for i in range(len(df2)):
    B = df2.loc[i]
    a=0.005
    ax.scatter(B[1],B[2],c='black')
#    ax.text(B[1]-0.001,B[2]-0.001,txt[i],fontsize=17)
    if i == 0:
        ax.text(B[1]-0.0003,B[2]-0.0005,txt[i],fontsize=17)
    elif i == 3:
        ax.text(B[1]-0.0003,B[2]+0.0003,txt[i],fontsize=17)
    else:
        ax.text(B[1]-0.0002,B[2]-0.0005,txt[i],fontsize=17)

for row in df1.itertuples():
    olon = df2[df2.mmsi==row.om]['lon'].values[0]
    olat = df2[df2.mmsi==row.om]['lat'].values[0]
    tlon = df2[df2.mmsi==row.tm]['lon'].values[0]
    tlat = df2[df2.mmsi==row.tm]['lat'].values[0]
    dx = tlon - olon
    dy = tlat - olat
    quiver(tlon,tlat,-dx,-dy, angles='xy', scale=1.03, scale_units='xy', width=0.008)
    
#ax.set_xlim(122.109,122.131)
#ax.set_ylim(29.835,29.855)
ax.set_xlim(122.13,122.16)
ax.set_ylim(29.913,29.92)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.get_xaxis().get_major_formatter().set_scientific(False)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False    

ax.set_xlabel('经度(°)',fontsize=17)
ax.set_ylabel('纬度(°)',fontsize=17) 
plt.tick_params(labelsize=17)
plt.show()

plot_pos(df2)