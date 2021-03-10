import pypsr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

def heaviside(value,threshold=0.7):
    if value >= threshold:
        return 1
    else:
        return 0

input_file = r'/home/mty/conflictNUM20181024.csv'
df = pd.read_csv(input_file)

origin_info = list(df.iloc[:, 1])
new_info = []
for index, row in enumerate(origin_info):
    if index%2 == 0:
        new_info.append(row)
warnings.filterwarnings("error")
reconstruct_list = pypsr.reconstruct(new_info, 8, 6)
# for Threshold in np.arange(0.81, 1.00, 0.01):
#     threshold = round(Threshold, 2)
threshold = 0.80
k_set = np.zeros(len(reconstruct_list))
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
# np.save('K_set('+str(threshold)+').npy', K_set)
np.save('K_set(24,8,0.80).npy', K_set)
    # print(K_set)
