from obspy import read
import numpy as np
import random
import os
from p_time import *
import matplotlib.pyplot as plt

threshold = 25
time_window = 500

PATH = ['../data57/']
def eachFile(filepath):

    magn = []
    for labelnanme in filepath:
        pathDir = os.listdir(labelnanme)
        tag = 0
        if labelnanme == '../data57/':
            for allDir in pathDir:
                # print(tag)
                if tag % 200 == 0:
                    print(tag, allDir[:-3])
                # print(allDir)
                child = os.path.join('%s%s' % (labelnanme, allDir))
                tag = tag + 1
                st = read(str(child))
                if (tag-1) % 3 == 0:
                    det_longitude = st[0].stats.knet.evla - st[0].stats.knet.stla
                    det_latitude = st[0].stats.knet.evlo - st[0].stats.knet.stlo


                if np.sqrt(det_longitude * det_longitude + det_latitude * det_latitude) > 2:
                    continue

                if tag % 3 == 0:
                    magn.append([st[0].stats.knet.mag, allDir[:-3]])


    return np.array(magn)

magn = eachFile(PATH)
np.save("event_magn57.npy",magn)
print(np.shape(magn))
