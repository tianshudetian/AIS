# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:07:12 2020

@author: dn4
"""
import xlrd
import numpy as np
file = r'C:/Users/dn4/Desktop/HiMCM/HiMCM2020ProblemB_ThreatenedPlantsData.xlsx'
wb = xlrd.open_workbook(file)
sh = wb.sheet_by_name('ThreatenedPlantsData')
dataTem = []
for i in range(48):
    dd = sh.row_values(i+1)[1:]
    dataTem.append(dd)
df = np.array(dataTem)