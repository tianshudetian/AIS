import requests
import json
import re
chuan = 413402430
headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
                }
session = requests.Session()
main_url = 'http://www.shipxy.com/'  # 推测对该url发起请求会产生cookie
session.get(main_url, headers=headers)
url = 'http://www.shipxy.com/ship/GetShip'
params = {
         'mmsi': chuan,
         }
page_text = session.get(url, headers=headers, params=params).json()
a=1