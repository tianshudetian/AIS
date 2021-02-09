# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:52:00 2020

@author: dn4
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
#C:\Users\dn4\Desktop\HiMCM\data.csv
file = r'C:/Users/dn4/Desktop/HiMCM/data.csv'
df = pd.read_csv(file,sep=',',header=None)
##计算项目持续时间
C_time = []#所有项目的持续时间,即
C_cost = []
for i in range(df.shape[0]):
    temp1 = df.iloc[i,3:]#临时文件
    cost_i_all = temp1.sum()#项目i的总花费
    C_cost.append(cost_i_all)
    for j in range(temp1.shape[0]):
        if temp1.iloc[j] > 0:
            d = j+1#d达最大时，即为当前项目的总持续时间
    C_time.append(d)
##总花费
cost_all = df.iloc[:,3:].sum().sum()
##收益数学期望
C_time_norm = minmax_scale(C_time)
C_cost_norm = minmax_scale(C_cost)
temp2 = pd.DataFrame(np.stack((C_time_norm,C_cost_norm))).T
temp3 = df.iloc[:,0:3]
temp4 = temp3.join(temp2, lsuffix='_left', rsuffix='_right')
temp4.rename(columns={'0_left': 'B', '1_left': 'U',2:'P','0_right':'C_time','1_right':'C_cost'},inplace=True)
E = temp4.apply(lambda x:(x['B']+x['U']-x['C_time']-x['C_cost'])*x['P']-(x['C_time']+x['C_cost'])*(1-x['P']),axis=1)#收益数学期望
E_norm = minmax_scale(E)
#x1 = df.iloc
#for k in range(df.shape[0]):
#    temp2 = df.iloc[k,:]
    
#%%
#参数设置
lambda1 = 1
lambda2 = 1
alpha = 0.5
beta = -1
#%%function
t_end = x[:,:] + C_time#计算所有项目的结束时间
#        ind = t_end.argmax(axis=1)#返回最后进行保护的植物物种
###求总持续时间
T = np.amax(t_end,axis=1)#返回最大时间  
###求解筹款难度
S = []#标准差
L = []#损失
for j in range(x.shape[0]):
    timetable = x[j]#提取一次项目安排
    budget = np.zeros(T[j])#T[j]为项目总持续时间
    loss = []#当前项目安排的总损失
    for i in range(len(timetable)):#i=1,2,…,48
        ti = timetable[i]#提取项目i的开始时间
        cost_i = df.iloc[i,3:]#项目i的cost
        loss_i = E_norm[i] - E_norm[i]**(alpha*(ti+beta))#项目i的损失
        loss.append(loss_i)
        for tt in range(cost_i.shape[0]):
            budget[ti+tt] = cost_i.iloc[tt]#项目i在ti+tt年时的cost
    s = np.std(budget,ddof=1)#求标准差，自由度为 N-ddof
    S.append(s)
    l = sum(loss)
    L.append(l)
M = lambda1*(cost_all/T)+lambda2*S#凑款难度
#目标函数
f1 = T
f2 = L
f3 = M

g1 = (np.zeros((x.shape,1)))**2 - 1e-5
        
#%%problem
from pymoo.model.problem import FunctionalProblem
from pymoo.model.problem import ConstraintsAsPenaltyProblem

problem = FunctionalProblem(2,
                            objs,
                            xl=-2,
                            xu=2,
                            constr_ieq=constr_ieq,
                            constr_eq=constr_eq
                            )

problem = ConstraintsAsPenaltyProblem(problem, penalty=1e6)
#%%optimize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize

algorithm = NSGA2(pop_size=40)

res = minimize(problem,
               algorithm,
               seed=1)
#%%visualize
from pymoo.visualization.scatter import Scatter

plot = Scatter(title = "Objective Space")
plot.add(res.F)
plot.show()