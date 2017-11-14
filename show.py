
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import  pylab
import os
from sklearn.preprocessing import OneHotEncoder


data=np.load('data.npy')
label=np.load('magn.npy')
dis=np.load('dis.npy')

for i in range(1000):
    plt.plot(data[i,0,:,1])
    plt.show()
    print(dis[i,:])