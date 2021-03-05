import pandas as pd
import numpy as np
from time import sleep
from scipy.interpolate import CubicSpline
def diff_over_threshold(list):
    Indexs = []
    for index, element in enumerate(list):
        if index >= 1:
            diff = element - list[index-1]
            if diff > 300:
                Indexs.append(index)
    return Indexs

def cubic(x_list,y_list):
    x = x_list - x_list[0]
    y = y_list - y_list[0]
    cs = CubicSpline(x, y)
    x_new = np.arange(0, x_list[-1]+1, 2)
    y_new = cs(x_new)
    return x_new+x_list[0], y_new+y_list[0]

or_df = pd.read_csv(r'/home/mty/data/dynamic/20181001.csv')
df = or_df.iloc[:10000, :]
# df.sort_values(by='timestamp', inplace=True)
latRange = [29.55, 30.1]
lonRange = [121.9, 122.45]
new_df = df.loc[(df['SOG'] >= 2.0) & (df['lon'] >= lonRange[0]) & (df['lon'] <= lonRange[-1]) & (df['lat'] >= latRange[0]) &
                       (df['lat'] <= latRange[-1])]
MMSIs = new_df['MMSI'].unique()
print('begin!!!')
for MMSI in MMSIs:
    print('MMSI: '+str(MMSI))
    tem = new_df[new_df['MMSI'].isin([MMSI])]
    temp = tem.drop_duplicates('timestamp', keep='first')
    time = temp['timestamp'].values
    Indexs = diff_over_threshold(time)
    Indexs.insert(0, 0)
    Indexs.append(len(time))
    new_indexs = list(set(Indexs))
    for i, index in enumerate(new_indexs):
        if i >= 1:
            if index - new_indexs[i-1] >= 10:
                index1 = new_indexs[i-1]
                index2 = index+1
                input_x = time[index1:index2]
                cubic_list = ['lat', 'lon', 'COG', 'SOG']
                new_temp = pd.DataFrame(columns=['MMSI'])
                new_temp['MMSI '] = MMSI
                for j, str_name in enumerate(cubic_list):
                    input_y = temp[str_name].values[index1:index2]
                    x_new, y_new = cubic(input_x, input_y)
                    new_temp[str_name] = y_new
                    print('j: '+str(j))
                    if j == 0:
                        new_temp['timestamp'] = x_new
                        new_temp['Length'] = temp['length'].iloc[0]
                print(new_temp)

test=1