from obspy import read
import numpy as np
import random
import os
from p_time import *
import matplotlib.pyplot as plt

threshold=25
time_window=500

PATH = ['../data57/']
def eachFile(filepath):
    data = []
    magn=[]
    dis=[]
    for labelnanme in filepath:
        pathDir = os.listdir(labelnanme)
        tag=0
        tempdata = []
        tempdata_f = []
        tempdata_n = []
        if labelnanme=='../data57/':
            for allDir in pathDir:

                if tag%200==0:
                    print(tag)
                    print(np.shape(data))
                child = os.path.join('%s%s' % (labelnanme, allDir))
                tag = tag + 1
                st = read(str(child))

                # tr_filt = st.copy()
                # tr_filt.filter('lowpass', freq=2.0, corners=2, zerophase=True)
                # f = tr_filt[0].data - tr_filt[0].data.mean()
                f = st[0].data - st[0].data.mean()

                if (tag-1)%3 == 0:
                    det_longitude = st[0].stats.knet.evla - st[0].stats.knet.stla
                    det_latitude = st[0].stats.knet.evlo - st[0].stats.knet.stlo
                    average_st = moving_average(st[0].data,10,'simple')
                    index = find_index(average_st,threshold)

                    if index - 50<0:
                        left = 0
                        right = 500
                    else:
                        left = index - 50
                        right = left + time_window

                if np.sqrt(det_longitude * det_longitude + det_latitude * det_latitude) > 2:
                    continue
                # print(f[left :right].shape)
                tempdata.append(f[left :right])
                # tempdata_n.append(allDir)
                # tempdata_f.append(f)                 # 测试画图用

                if tag % 3 == 0:
                    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    #
                    # plt.figure(1)
                    # plt.subplot(321)
                    # plt.title("%f| %f | %d| %d| %d"
                    #           %(np.sqrt(det_longitude * det_longitude + det_latitude * det_latitude)
                    #             ,st[0].stats.knet.mag, index,left ,right))
                    # plt.plot(tempdata[0])
                    # plt.subplot(323)
                    # plt.plot(tempdata[1])
                    # plt.subplot(325)
                    # plt.plot(tempdata[2])
                    # plt.subplot(322)
                    # plt.title(tempdata_n[0])
                    # plt.plot(tempdata_f[0])
                    # plt.subplot(324)
                    # plt.title(tempdata_n[1])
                    # plt.plot(tempdata_f[1])
                    # plt.subplot(326)
                    # plt.title(tempdata_n[2])
                    # plt.plot(tempdata_f[2])
                    # plt.show()


                    magn.append(st[0].stats.knet.mag)
                    dis.append([np.sqrt(det_longitude*det_longitude+det_latitude*det_latitude),np.abs(det_longitude),np.abs(det_latitude),st[0].stats.knet.stel])
                    tempdata = np.array(tempdata)
                    tempdata = np.transpose(tempdata)
                    tempdata.tolist()
                    data.append([tempdata])
                    tempdata = []
                    tempdata_f = []
                    tempdata_n = []

    return np.array(data),np.array(magn),np.array(dis)

data,magn,dis=eachFile(PATH)
np.save("n_data57.npy",data)
np.save("n_magn57.npy",magn)
np.save("n_dis57.npy",dis)
print(np.shape(data))
print(np.shape(magn))
print(np.shape(dis))
