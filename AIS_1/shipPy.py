# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:04:22 2020

@author: dn4
"""

import requests
import pandas as pd

inputFile = r"" ##输入文件名
outputFile = r""##输出文件名
df = pd.read_csv(inputFile,header=True)
res = []#结果集
for row in df.itertuples():
    MMSI = row.MMSI
#MMSI = '413561000'
    dic={
         'f':'srch',
         'kw':MMSI
          }
    #'http://searchv3.shipxy.com/shipdata/search3.ashx?f=srch&kw=413561000'
    rq=requests.get("http://searchv3.shipxy.com/shipdata/search3.ashx",params=dic)
    if rq.status_code==200:
        try:
            result_json=rq.json()
            shipType = result_json['ship'][0]['t']
            res.append([MMSI,shipType])
        except:
            print(str(MMSI)+"无对应数据")
    else:
        print(str(MMSI)+"无对应数据")

result = pd.DataFrame(res,columns = {'a','b'})
result.rename(columns={'a':'MMSI','b':'type'},inplace=True)
result.to_csv(outputFile,sep=',')