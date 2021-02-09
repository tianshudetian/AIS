# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:26:57 2019

@author: dn4

Purpose: deal with the static ship information

file number: two
"""

import pandas as pd

inputfile='D:/处理数据/舟山2017年7月静态数据.csv'
#‪D:\论文数据\static\201810-12StaticData.csv
#‪D:\处理数据\舟山2017年7月静态数据.csv
outputfile='D:/论文数据/static/舟山201707.csv'


reader = pd.read_csv(inputfile,header=0,encoding='gbk',engine='python',\
                     iterator=True,usecols=['~MMSI','Length','Ship Type'],\
                     error_bad_lines=False)
#选定所需的列
loop = True
chunksize = 1000000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunksize)
        chunk = chunk.rename(columns={'~MMSI':'MMSI','Length':'length','Ship Type':'type'})
        chunk1 = chunk[chunk['type'].isin([70])]
        chunk2 = chunk[chunk['type'].isin([80])]
        chunk = pd.concat([chunk1,chunk2])
        chunk.dropna(axis=0,how='any',inplace=True)
        chunk['MMSI']=chunk['MMSI'].astype('int')#转换数据格式
        chunk = chunk.loc[(chunk.length<450)&(chunk.length>40) & (chunk.MMSI>=000000000) & (chunk.MMSI<=999999999)]
        #块中去重
        chunk = chunk.drop_duplicates(['MMSI'])#舍弃MMSI重复的行,随机保留一个船长
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
chunks = pd.concat(chunks,ignore_index=True)
#全局去重        
chunks = chunks.drop_duplicates(['MMSI'])#舍弃MMSI重复的行,随机保留一个船长
chunks.to_csv(outputfile,sep=',',header=True,index=0)
