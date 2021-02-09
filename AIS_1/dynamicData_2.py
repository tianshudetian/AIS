# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:29:21 2019

@author: dn4

Purpose: deal with the dynamic ship information

file number: one_2
"""

import pandas as pd
import time

inputfile='D:/处理数据/decode20181021.csv'
outputfile='D:/数据/20181021.csv'


reader = pd.read_csv(inputfile,encoding='gbk',iterator=True,engine='python',header=0, error_bad_lines=False)
#   D:\论文数据\dynamic\处理数据\Decode20181017.csv
#   文件名需手打
#   iterator, chunksize进行分块,推荐chunksize为100000000 
#   encoding, engine处理文件名中的中文
#   error_bad_lines跳过报错行
#%%
loop = True
chunksize = 1000000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunksize)
        chunk['Nmea Prefix String']=chunk['Nmea Prefix String'].fillna(method='ffill')
        chunk.dropna(axis=0,how='any',inplace=True)
        #   axis：0-行操作（默认），1-列操作 
        #   how：any-只要有空值就删除（默认），all-全部为空值才删除 
        #   inplace：False-返回新的数据集（默认），True-在原数据集上操作
#        for row in chunk.itertuples():
#            if len(row) >8:
#                chunk = chunk.drop(labels=row.Index,axis=0)
#        chunk.columns=('MMSI','SOG','lon','lat','COG','HDG','UTC')
        chunk=chunk.rename(columns={'~MMSI':'MMSI','Speed Over Ground (SOG)':'SOG','Longitude':'lon','Latitude':'lat','Course Over Ground (COG)':'COG','Nmea Prefix String':'UTC'})
        #  更改列名
        chunk['date_parsed'] = pd.to_datetime(chunk['UTC'], format = "$OSNT,1,%Y,%m,%d,%H,%M,%S\n",errors = 'coerce')
        #   有部分数据为NaT
        chunk = chunk[~chunk.date_parsed.isnull()]
        #   剔除为NaT的数据
        chunk['timestamp'] = chunk['date_parsed'].apply(lambda x:time.mktime(x.timetuple()))
        #   时间转时间戳
        chunk.drop(labels=['UTC','date_parsed'],axis=1,inplace=True)
        #   删除原时间与解析时间
        chunk['MMSI']=chunk['MMSI'].astype('int')#转换数据格式
        chunk['SOG']=chunk['SOG'].astype('float')#原数据为object格式
        chunk['lon']=chunk['lon'].astype('float')
        chunk['lat']=chunk['lat'].astype('float')
        chunk['COG']=chunk['COG'].astype('float')
        chunk=chunk.loc[(chunk.lat>=29.8) & (chunk.lat<=30) & (chunk.lon>=122.1) & (chunk.lon<=122.25) & (chunk.COG>=0) & (chunk.COG<=360) & (chunk.SOG>=3) & (chunk.MMSI>=000000000) & (chunk.MMSI<=999999999)]
        #   筛选范围如下:
        #   纬度范围(29.8~30)=(29°48′-30°00′)
        #   经度范围(122.1~122.25)=(122°06′-122°15′)
        #   COG范围(0~360°)
        #   MMSI范围(000000000~999999999)
        #   船舶航速范围>=3kn
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
df = pd.concat(chunks, ignore_index=True)
df.to_csv(outputfile, sep=',', header=True,index=0)
print(chunk.head(10))