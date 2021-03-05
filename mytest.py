import pypsr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = r'/home/mty/data/11.24结果/df/20181001df6.csv'

df = pd.read_csv(input_file)
# Timestamps = df['time']
# newTime = np.arange(Timestamps.min(), Timestamps.max()+2, 2)
# count_list = []
# for newt in newTime:
#     count_list.append(df[df['time'].isin([newt])].shape[0])
a=1