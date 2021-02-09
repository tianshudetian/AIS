import pandas as pd
from hilbert import decode, encode
def positionConvert(data):
    tem = data.copy()
    tem['newLat'] = tem['lat'].apply(lambda x: round((x-lat_rang[0])/(lat_rang[1]-lat_rang[0])*pow(2, Ganularity)))
    tem['newLon'] = tem['lon'].apply(lambda x: round((x-lon_rang[0])/(lon_rang[1]-lon_rang[0])*pow(2, Ganularity)))
    return tem

def hilbertEncode(data):
    tem = data.copy()
    pos = tem[['newLat', 'newLon']].values
    H = encode(pos, Dimension, Ganularity)
    tem['H'] = H
    return tem

# if __name__=="__main__":
df = pd.read_csv(r'/home/mty/data/processedData/20181024Processed.csv')
Timestamps = list(set(df['timestamp']))
# for timestamp in Timestamps:
timestamp = Timestamps[10]
temp = df[df['timestamp'].isin([timestamp])]
Ganularity = 10
Dimension = 2
lat_rang = [29.55, 30.1]
lon_rang = [121.9, 122.45]
newTemp = positionConvert(temp)
H = hilbertEncode(newTemp)
b=1
a=1
