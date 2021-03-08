import pypsr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

def heaviside(value,threshold=0.8):
    if value >= threshold:
        return 1
    else:
        return 0

input_file = r'/home/mty/conflictNUM.csv'
df = pd.read_csv(input_file)
origin_info = list(df.iloc[:, 1])
reconstruct_list = pypsr.reconstruct(origin_info, 12, 6)
k_set=np.zeros(len(reconstruct_list))
threshold = 0.8
warnings.filterwarnings("error")
for index1, vector1 in enumerate(reconstruct_list):
    tem_list = reconstruct_list[index1:, :]
    if len(tem_list) > 1:
        for index2, vector2 in enumerate(tem_list):
            try:
                correlation_coefficient = np.corrcoef(vector1, vector2)
                D = heaviside(correlation_coefficient[0][1], threshold)
                if D == 1:
                    k_set[index1] = k_set[index1] + D
                    k_set[index2+index1+1] = k_set[index2+index1+1] + D
            except:
                pass
K_set = np.array(k_set)
print(K_set)
np.save('K_set.npy', K_set)