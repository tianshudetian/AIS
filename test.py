# -*- coding: utf-8 -*-


import requests
import json
import pandas as pd

inputFile = r"C:/Users/李尚泽/Desktop/66666.csv"  ##输入文件名
outputFile = r"C:\\Users\\李尚泽\\Desktop\\777.csv"  ##输出文件名
df = pd.read_csv(inputFile, header=None, encoding='gbk', engine='python')
df.columns = ['MMSI']
res = []  # 结果集
url1 = "http://searchv3.shipxy.com/shipdata/search3.ashx"
url2 = "http://www.shipxy.com/ship/GetShip"
for row in df.itertuples():
    MMSI = row.MMSI
    print('MMSI: ' + str(MMSI))
    # MMSI = '413050330'
    dic = {
        'f': 'srch',
        'kw': MMSI
    }

    # 'http://searchv3.shipxy.com/shipdata/search3.ashx?f=srch&kw=413561000'
    try:

        rq1 = requests.get(url1, params=dic, timeout=0.5)

        if rq1.status_code == 200:
            try:
                result_json = rq1.json()
                shipType = result_json['ship'][0]['t']
                #                data={
                #                        'mmsi':MMSI
                #                        }
                #                rq2=requests.get(url2,data=data,headers={'Host': 'www.shipxy.com',
                #                                                         'Connection':'keep-alive',
                #                                                         'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8'
                #                                                         },timeout=0.5)
                res.append([MMSI, shipType])
            except:
                shipType = 999
                res.append([MMSI, shipType])
        else:
            shipType = 999
            res.append([MMSI, shipType])

    except:
        print('Error')

result = pd.DataFrame(res, columns={'a', 'b'})
result.columns = ['MMSI', 'TYPE']
# result.rename(columns={'a':'MMSI','b':'type'},inplace=True)
result.to_csv(outputFile, sep=',')
