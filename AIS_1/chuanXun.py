# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:20:42 2020

@author: dn4
"""

import requests


url = 'http://www.shipxy.com/ship/GetShip'

MMSI = 636013778
data = {'mmsi':MMSI}

rq = requests.post(url,data=data)
print(rq.text)
print(rq)
