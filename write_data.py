from obspy import read
import numpy as np
import random
import os
from p_time import *
import matplotlib.pyplot as plt

threshold=25
time_window=1000

PATH = ['../data/']
def eachFile(filepath):
    data = []
    magn=[]
    dis=[]
    for labelnanme in filepath:
        pathDir = os.listdir(labelnanme)
        tag=0
        tempdata=[]
        if labelnanme=='../data/':
            for allDir in pathDir:
                if tag%100==0:
                    print(tag)
                    # print(np.shape(data))
                child = os.path.join('%s%s' % (labelnanme, allDir))
                tag = tag + 1
                # print(allDir)
                st = read(str(child))
                average_st=moving_average(st[0].data,10,'simple')
                index=find_index(average_st,threshold)
                f = st[0].data - st[0].data.mean()
                det_longitude = st[0].stats.knet.evla - st[0].stats.knet.stla
                det_latitude = st[0].stats.knet.evlo - st[0].stats.knet.stlo
                if np.sqrt(det_longitude * det_longitude + det_latitude * det_latitude)>1:

                    continue

                tempdata.append(f[index:index + time_window])
                if tag % 3 == 0:

                    # # 测试画图
                    # plt.plot(f)
                    # temp_tag_list = np.zeros(len(st[0].data))
                    # temp_tag_list[index] = temp_tag_list[index + time_window] = f.max()
                    # plt.plot(temp_tag_list)
                    # plt.show()
                    # print(np.sqrt(det_longitude*det_longitude+det_latitude*det_latitude))
                    # #
                    print(allDir)
                    magn.append(st[0].stats.knet.mag)
                    dis.append([np.sqrt(det_longitude*det_longitude+det_latitude*det_latitude),np.abs(det_longitude),np.abs(det_latitude),st[0].stats.knet.stel])
                    tempdata = np.array(tempdata)
                    tempdata=np.transpose(tempdata)
                    tempdata.tolist()
                    data.append([tempdata])
                    tempdata = []
    return np.array(data),np.array(magn),np.array(dis)

data,magn,dis=eachFile(PATH)
np.save("data.npy",data)
np.save("magn.npy",magn)
np.save("dis.npy",dis)
print(np.shape(data))
print(np.shape(magn))
print(np.shape(dis))
