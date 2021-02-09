# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:16:26 2020

@author: dn4
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import xlrd
file = r'C:/Users/dn4/Desktop/HiMCM/HiMCM2020ProblemB_ThreatenedPlantsData.xlsx'
wb = xlrd.open_workbook(file)
sh = wb.sheet_by_name('ThreatenedPlantsData')
dataTem = []
for i in range(48):
    dd = sh.row_values(i+1)[1:]
    dataTem.append(dd)
df = pd.DataFrame(np.array(dataTem))
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
lambda1 = 0.35
lambda2 = 0.65
alpha = 0.4
beta = -1
#%%设置目标函数和约束函数
from pymoo.model.problem import Problem


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=48,
                         n_obj=3,
                         n_constr=1,
                         xl=np.ones(48,dtype=int),#生成 全1数组,表示起始于第一年
                         xu=np.full(48, fill_value=50),
                         type_var=np.int)#生成全50数组，表示结束于第一年
                         
    def _evaluate(self, x, out, *args, **kwargs):
        #计算所有项目的结束时间
        tend = []
        for m in range(x.shape[1]):
            t = x[:,m] + C_time[m] - 1
            tend.append(t)
        t_end = np.array(pd.DataFrame(tend).T)
#        ind = t_end.argmax(axis=1)#返回最后进行保护的植物物种
        ###求总持续时间
        T = np.amax(t_end,axis=1)#返回最大时间
#        print('时间安排'+str(x))
#        print('最大时间'+str(T))
        ###求解筹款难度
        S = []#标准差
        L = []#损失
        for j in range(x.shape[0]):
            timetable = x[j]#提取一次项目安排
            budget = np.zeros(T[j])#T[j]为项目总持续时间
            loss = []#当前项目安排的总损失
            for i in range(len(timetable)):#i=0,1,2,…,47
                ti = timetable[i]#提取项目i的开始时间
                cost_i = df.iloc[i,3:]#项目i的cost
                loss_i = E_norm[i] - E_norm[i]**(alpha*(ti+beta))#项目i的损失
                loss.append(loss_i)
                for tt in range(len(cost_i)):
                    if cost_i.iloc[tt] != 0:
                        budget[ti+tt-1] = budget[ti+tt-1] + cost_i.iloc[tt]#ti+tt年时的cost
            s = np.std(budget,ddof=1)#求标准差，自由度为 N-ddof
            S.append(s)
            l = sum(loss)
            L.append(l)
        M = np.dot(lambda1,(cost_all/T))+np.dot(lambda2,S)#凑款难度
#        print('筹款难度'+str(M))
                
        g1 = (np.zeros((x.shape[0],1)))**2 - 1e-5

        out["F"] = np.column_stack([T, L, M])
        out["G"] = g1

#%%使用NSGA2求解pareto前沿
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling

method = get_algorithm("nsga2",
                       pop_size=100,
                       sampling=get_sampling("int_random"),
                       crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                       mutation=get_mutation("int_pm", eta=3.0),
                       eliminate_duplicates=True,
                       )

from pymoo.optimize import minimize
#from pymoo.visualization.scatter import Scatter

res = minimize(MyProblem(),
               method,
               termination=('n_gen', 40),
               seed=1,
               save_history=True)

#plot = Scatter(title = "Objective Space")
#plot.add(res.F, color="red")
#plot.show()
pareto_front =res.F
T = pareto_front[:,0]
L = pareto_front[:,1]
M = pareto_front[:,2]
#%%可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(T,L,M,c='r')

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax.set_xlabel('T',fontsize=12)
ax.set_ylabel('L',fontsize=12) 
ax.set_zlabel('M',fontsize=12) 
plt.tick_params(labelsize=12)
plt.show()
#%%可视化
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.scatter(T,L)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax1.set_xlabel('T',fontsize=12)
ax1.set_ylabel('L',fontsize=12)
plt.tick_params(labelsize=12)

fig = plt.figure()
ax2 = fig.add_subplot(111)
plt.scatter(T,M)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax2.set_xlabel('T',fontsize=12)
ax2.set_ylabel('M',fontsize=12)
plt.tick_params(labelsize=12)

fig = plt.figure()
ax3 = fig.add_subplot(111)
plt.scatter(L,M)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.set_xlabel('L',fontsize=12)
ax3.set_ylabel('M',fontsize=12)
plt.tick_params(labelsize=12)

#%%topsis最优化
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


weights = [0.3,0.3,0.4]#权重矩阵
answer1 = pareto_front.T     #读取文件
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
result = pd.DataFrame(pareto_front,columns = {'T','L','M'})
result['score'] = answer5        #计算得分
#将得分输出到excel表格中
result.to_csv(r'C:/Users/dn4/Desktop/HiMCM/result.csv')
#%%筹款计划
schedule = res.X
ind = result[result.loc[:,'score'].isin([result['score'].max()])].index
Schedule = schedule[ind]
print("项目开始年份："+str(Schedule))
