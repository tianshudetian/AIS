# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:06:12 2020

@author: dn4
"""

import glob
from os import system
fileList = glob.glob(r'c:\Users\dn4\Desktop\CSSD.pdf')
for f in fileList:
  system('pdftops -eps {0}'.format(f))