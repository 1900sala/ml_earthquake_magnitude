# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

input_vec_size = lstm_size = 10  # 输入向量的维度
time_step_size = 149  # 循环层长度

train_size = 25000
batch_size = 50
data1 = np.load('n_nn_taoc_dataud.npy')
data2 = np.load('n_nn_taoc_data57ud.npy')
data = np.concatenate((data1, data2), axis=0)
label1 = np.load('n_magn.npy')
label2 = np.load('n_magn57.npy')
label = np.concatenate((label1, label2), axis=0)
event_label1 = np.load('event_magn.npy')
event_label2 = np.load('event_magn57.npy')
event_label = np.concatenate((event_label1, event_label2), axis=0)
w = np.load('w.npy')
label = label.reshape(len(label), 1)
dis1 = np.load('n_dis.npy')
dis2 = np.load('n_dis57.npy')
dis = np.concatenate((dis1, dis2), axis=0)


f_threshold = np.where((label > 4))
data =  data[f_threshold[0]]
label = label[f_threshold[0]]
event_label = event_label[f_threshold[0]]
#标准化
for i in range(len(data)):
    data[i, :] = data[i, :]*w
for i in range(149):
    data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()



def fc_layer(bottom, name, shape=None):
    with tf.variable_scope(name) as scope:
        stddev = 0.05
        wd = 5e-4
        weight = variable_with_weight_decay(shape, stddev, wd)
        initB = tf.constant_initializer(0.0)
        bias = tf.get_variable(name='bias', shape=shape[1], initializer=initB)
        fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
        if name == 'output':
            return fc, weight
        else:
            relu = tf.nn.relu(fc)
    return relu, weight


def variable_with_weight_decay(shape, stddev, wd):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)
    return var


def train_test_split(data, dis, label, test_size=0.1):
    data_len = len(data)
    index = np.array(range(data_len))
    random.shuffle(index)
    data_batch_loc = index[ : (data_len+1)*(1-test_size)]
    ret = index[(data_len+1)*(1-test_size) : ]
    return data[data_batch_loc], dis[data_batch_loc], label[data_batch_loc], data[ret], dis[ret], label[ret]\
        ,data_batch_loc,ret


class shock_data(object):
    def __init__(self, inputs, dis, labels):
        self.inputs = inputs
        self.dis = dis
        self.labels = labels

    def batch(self, size):
        data_len = len(self.inputs)
        index = np.array(range(data_len))
        random.shuffle(index)
        data_batch_loc = index[: size]
        a = self.inputs[data_batch_loc]
        b = self.dis[data_batch_loc]
        c = self.labels[data_batch_loc]
        # print(np.shape(b))
        # print(b)
        return a, b, c


class shock_all(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))




x_train, dis_train, y_train, x_test, dis_test, y_test, data_batch_loc, ret\
    = train_test_split(data, dis, label, test_size=0.2)

# n_threshold1 = np.where((y_train <6)&(y_train > 5.5))
# n_threshold2 = np.where((y_train <4)&(y_train > 3.5))
#
# x_train =  np.concatenate((x_train, x_train[n_threshold1[0]], x_train[n_threshold2[0]]
#                         , x_train[n_threshold1[0]], x_train[n_threshold2[0]]), axis=0)
# y_train =  np.concatenate((y_train, y_train[n_threshold1[0]], y_train[n_threshold2[0]]
#                          , y_train[n_threshold1[0]], y_train[n_threshold2[0]]), axis=0)
# dis_train = np.concatenate((dis_train, dis_train[n_threshold1[0]], dis_train[n_threshold2[0]]
#                       , dis_train[n_threshold1[0]], dis_train[n_threshold2[0]]), axis=0)
train = shock_data(x_train, dis_train, y_train)
test = shock_data(x_test, dis_test, y_test)
shock = shock_all(train, test)
print(len(shock.train.inputs),len(shock.test.inputs))


X = tf.placeholder("float", [None, 149])
Y = tf.placeholder("float", [None, 1])
keep_prob = tf.placeholder("float")

# get lstm_size and output 10 labels

h_fc1, W_fc1 = fc_layer(X, 'fc1', shape=[149, 256])
h_fc2, W_fc2 = fc_layer(h_fc1, 'fc2', shape=[256, 256])
h_fc2 = tf.nn.dropout(h_fc2, keep_prob)
h_fc3, W_fc3 = fc_layer(h_fc2, 'fc3', shape=[256, 256])
h_fc4, W_fc4 = fc_layer(h_fc3, 'fc4', shape=[256, 256])
h_fc4 = tf.nn.dropout(h_fc4, keep_prob)
h_fc5, W_fc5 = fc_layer(h_fc4, 'fc5', shape=[256, 256])
h_fc6, W_fc6 = fc_layer(h_fc5, 'fc6', shape=[256, 256])
h_fc6 = tf.nn.dropout(h_fc6, keep_prob)
h_fc7, W_fc7 = fc_layer(h_fc6, 'fc7', shape=[256, 128])
y, W_fc8 = fc_layer(h_fc7, 'output', shape=[128, 1])

lam = 0.0001
regularizers = lam*(tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(W_fc4) +
                tf.nn.l2_loss(W_fc5) + tf.nn.l2_loss(W_fc6) + tf.nn.l2_loss(W_fc7) + tf.nn.l2_loss(W_fc8) ) \
               # + \
               # tf.contrib.layers.l1_regularizer(lam)(W_fc1)+ tf.contrib.layers.l1_regularizer(lam)(W_fc2)+ \
               # tf.contrib.layers.l1_regularizer(lam)(W_fc3)+ tf.contrib.layers.l1_regularizer(lam)(W_fc4)+\
               # tf.contrib.layers.l1_regularizer(lam)(W_fc5)+ tf.contrib.layers.l1_regularizer(lam)(W_fc6)+\
               # tf.contrib.layers.l1_regularizer(lam)(W_fc7)+ tf.contrib.layers.l1_regularizer(lam)(W_fc8)

mse = tf.reduce_mean(tf.square(y - Y))
loss = mse + regularizers
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

train_l = []
test_l = []

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(train_size):
        batch = shock.train.batch(50)

        if i % 500 == 0:
            mse_val = mse.eval(feed_dict={X: shock.train.inputs, Y: shock.train.labels, keep_prob: 1})
            regularizers1 = regularizers.eval(feed_dict={X: shock.train.inputs, Y: shock.train.labels, keep_prob: 1})
            print("step %d/%d, training mse_val %g" % (i, train_size, mse_val))
            print(regularizers1)
            mse_val_test = mse.eval(feed_dict={X: shock.test.inputs, Y: shock.test.labels, keep_prob: 1})
            print("test mse_val %g" % mse_val_test)
            train_l.append(mse_val)
            test_l.append(mse_val_test)

        train_op.run(feed_dict={X: batch[0], Y: batch[2], keep_prob: 0.5})

    plt.plot(train_l[4:], c='red', label='train')
    plt.plot(test_l[4:], c='black', label='test')
    plt.ylabel(' acc')
    plt.legend()
    plt.show()

    p1 = y.eval(feed_dict={X: shock.test.inputs, Y: shock.test.labels, keep_prob: 1})
    print(p1.shape)
    y1 = shock.test.labels
    print("test mse_val %g" % mse.eval(
        feed_dict={X: shock.test.inputs, Y: shock.test.labels, keep_prob: 1}))
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y1, c='red', label='y')
    plt.plot(p1, c='black', label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()

    p2 = y.eval(feed_dict={X: shock.train.inputs, Y: shock.train.labels, keep_prob: 1})
    y2 = shock.train.labels
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y2, c='red', label='y')
    plt.plot(p2, c='black', label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()

tst = event_label[ret]
print(tst.shape, tst[:,1].shape, p1.shape )
p1 = p1.reshape(len(p1))
df = pd.DataFrame({'event':tst[:,1],'logtaoc':p1, 'mag':tst[:,0]})
df['event'] = df['event'].apply(lambda x: x[:-1] if len(x)==17 else x)
df['event'] = df['event'].apply(lambda x: x[-10:] )
df['mag'] = df['mag'].apply(lambda x: np.float(x) )
y = df.groupby(['event'])['logtaoc'].mean()
std = df.groupby(['event'])['logtaoc'].std()
x = df.groupby(['event'])['mag'].mean()
# c = []
# for i in range(len(std)):
#     c.append([y[i]-std[i], y[i]+std[i]])
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s",ls='none',fillstyle='none',ms=9,mew=1.3,color='r')
plt.show()

rmse = np.sqrt(((x-y)*(x-y)).mean())
print(rmse)