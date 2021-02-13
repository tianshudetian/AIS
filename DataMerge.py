import pandas as pd
import os
from multiprocessing import Pool
from AIS import AIScrawler

def getfile(inputpath):
    pathList = os.listdir(inputpath)
    inputFileLIst = []
    for i in range(0, len(pathList)):
        inputPath = os.path.join(inputpath, pathList[i])
        if os.path.isfile(inputPath):
            inputFileLIst.append(inputPath)
    return inputFileLIst

def datamerge(inputfile):
    print('input file: ' + str(inputfile))
    path = inputfile[-12:]
    outputFath = r'/home/mty/data/dynamic'
    outputfile = os.path.join(outputFath, path)
    print('output file: ' + str(outputfile))
    try:
        DynamicData = pd.read_csv(inputfile)
        MMSIs = list(set(DynamicData['MMSI']))
        newStatic = []
        failMatchStatic = []
        for MMSI in MMSIs:
            tem = StaticData[StaticData['MMSI'].isin([MMSI])]
            if tem.shape[0] >= 1:
                newStatic.append(tem)
            elif tem.shape[0] > 1:
                print('Search exception: ' + str(MMSI))
            else:
                failMatchStatic.append(MMSI)
        crawler = AIScrawler(failMatchStatic)
        crawler.requestShipxy()
        StaticAnother = crawler.newdata
        newStatic.append(StaticAnother)
        newStatic = pd.concat(newStatic, ignore_index=True)
        newData = pd.merge(DynamicData, newStatic, on='MMSI')
        newData.to_csv(outputfile, sep=',', header=True, index=0)
    except:
        print('Error with file: '+str(inputfile))

if __name__=="__main__":
    StaticData = pd.read_csv(r'/home/mty/data/FilteredStatic/201810-12StaticData.csv', engine='python', encoding='gbk')
    inputPath = r'/home/mty/data/dynamicNofilter'
    inputFileList = getfile(inputPath)
    with Pool(5) as p:
        newDataList = p.map(datamerge, inputFileList)
        p.close()
        p.join()