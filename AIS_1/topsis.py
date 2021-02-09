# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:23:56 2020

@author: dn4
"""

import numpy as np
#'‪C:\Users\dn4\AIS_1\pareto_front.npy'
#pareto_front = np.load(r'C:/Users/dn4/AIS_1/pareto_front.npy')
#weights = [0.4,0.3,0.3]

#%%
import pandas as pd

#极小型指标 -> 极大型指标
def dataDirection_1(datas):         
        return np.max(datas)-datas     #套公式

#中间型指标 -> 极大型指标
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M     #套公式
    return answer_datas
    
#区间型指标 -> 极大型指标
def dataDirection_3(datas, x_min, x_max):
    M = max(x_min - np.min(datas), np.max(datas) - x_max)
    answer_list = []
    for i in datas:
        if(i < x_min):
            answer_list.append(1 - (x_min-i) /M)      #套公式
        elif( x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max)/M)
    return np.array(answer_list)   

#加权
def temp1(datas,weights):
    tem = datas.copy()
    wDatas = (np.multiply(tem.T,weights)).T
    return wDatas

#正向化矩阵标准化
def temp2(datas):
    K = np.power(np.sum(pow(datas,2),axis =1),0.5)
    for i in range(0,K.size):
        for j in range(0,datas[i].size):
            datas[i,j] = datas[i,j] / K[i]      #套用矩阵标准化的公式
    return datas

#计算得分并归一化
def temp3(answer2):
    list_max = np.array([np.max(answer2[0,:]),np.max(answer2[1,:]),np.max(answer2[2,:])])  #获取每一列的最大值
    list_min = np.array([np.min(answer2[0,:]),np.min(answer2[1,:]),np.min(answer2[2,:])])  #获取每一列的最小值
    max_list = []       #存放第i个评价对象与最大值的距离
    min_list = []       #存放第i个评价对象与最小值的距离
    answer_list=[]      #存放评价对象的未归一化得分
    for k in range(0,np.size(answer2,axis = 1)):        #遍历每一列数据
        max_sum = 0
        min_sum = 0
        for q in range(0,3):                                #有三个指标
            max_sum += np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
            min_sum += np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
        max_list.append(pow(max_sum,0.5))
        min_list.append(pow(min_sum,0.5))
        answer_list.append(min_list[k]/ (min_list[k] + max_list[k]))    #套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)      #得分归一化
    return (answer / np.sum(answer))


def main():
    file = r'C:/Users/dn4/AIS_1/pareto_front.npy'
    df = np.load(file)
    weights = [0.3,0.3,0.4]#权重矩阵
    answer1 = df.T     #读取文件
    answer2 = []
    for i in range(answer1.shape[0]):
        answer = None
        if(i == 0):             #持续时间为极小型指标，越小越好
            answer = dataDirection_1(answer1[0])
        elif(i == 1):           #收益损失为为极小型指标
            answer = dataDirection_1(answer1[1]) 
        else:                     #极小型指标
            answer = dataDirection_1(answer1[2])
        answer2.append(answer)
    answer2 = np.array(answer2)         #将list转换为numpy数组
    answer3 = temp1(answer2,weights)
    answer4 = temp2(answer3)            #数组正向化
    answer5 = temp3(answer4)            #标准化处理
    data = pd.DataFrame(answer5)        #计算得分
    #将得分输出到excel表格中
    data.to_csv(r'C:/Users/dn4/Desktop/HiMCM/result.csv')
    return data

result = main()