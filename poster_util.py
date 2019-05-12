from obspy import read
import numpy as np
import random
import os
from p_time import *
import matplotlib.pyplot as plt

# n3_taoc_data = np.load('n3_taoc_data.npy')
# t_len = 300
# fft_ss_len = int(t_len / 2 - 1)
# x = np.array(range(1,fft_ss_len+1))/fft_ss_len*100
#
# for i in range(1200, 1201):
#     plt.figure(1)
#     plt.subplot(311)
#     plt.plot(n3_taoc_data[0, 0, :48]*x[0:48], color = 'r', linewidth = 5)
#     plt.subplot(312)
#     plt.plot(n3_taoc_data[0, 1, :48]*x[0:48], color = 'b', linewidth = 5)
#     plt.subplot(313)
#     plt.plot(n3_taoc_data[0, 2, :48]*x[0:48], color = 'g', linewidth = 5)
#     plt.show()
#


n_data = np.load('n_data.npy')


for i in range(1200, 1201):
    plt.figure(1)
    plt.subplot(311)
    plt.plot(n_data[i, 0, :, 0], color = 'r', linewidth = 2)
    plt.subplot(312)
    plt.plot(n_data[i, 0, :, 1], color = 'b', linewidth = 2)
    plt.subplot(313)
    plt.plot(n_data[i, 0, :, 2], color = 'g', linewidth = 2)
    plt.show()