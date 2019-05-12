# -*- coding: utf-8 -*-
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

input_vec_size = lstm_size = 10  # 输入向量的维度
time_step_size = 149  # 循环层长度

train_size = 18000
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

f_threshold = np.where((label >= 3) & (label < 8))
data = data[f_threshold[0]]
label = label[f_threshold[0]]

plt.subplot(1, 2, 1)
plt.hist(label, color='grey', bins=15)
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Number of records', fontsize=26)
plt.tick_params(labelsize=20)
plt.text(6, 4500, r"(a)", fontsize=18)
# plt.legend(fontsize=20)
# plt.title("Pre-adjustment magnitude distribution(a)", fontsize=26)
# plt.show()
event_label = event_label[f_threshold[0]]
dis = dis[f_threshold[0]]

n_threshold1 = np.where(label >= 5.6)
n_threshold2 = np.where(label <= 3.5)
print(len(n_threshold1[0]), len(n_threshold2[0]))
label = np.concatenate((label, label[n_threshold1[0]], label[n_threshold2[0]]
                        , label[n_threshold1[0]], label[n_threshold2[0]]), axis=0)
d1 = data[n_threshold1[0]] + np.random.random(data[n_threshold1[0]].shape) /10
d2 = data[n_threshold2[0]] + np.random.random(data[n_threshold2[0]].shape) /10
d3 = data[n_threshold1[0]] + np.random.random(data[n_threshold1[0]].shape) /10
d4 = data[n_threshold2[0]] + np.random.random(data[n_threshold2[0]].shape) /10

plt.subplot(1, 2, 2)
plt.hist(label, bins=15)
plt.hist(label, color='grey', bins=15)
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Number of records', fontsize=26)
plt.tick_params(labelsize=20)
plt.text(6, 6000, r"(b)", fontsize=18)
# plt.legend(fontsize=20)
# plt.title("Adjusted magnitude distribution(b)", fontsize=26)
# plt.savefig("pic/event_distribution.eps", format='eps')
plt.show()
data = np.concatenate((data, d1, d2, d3, d4), axis=0)
dis = np.concatenate((dis, dis[n_threshold1[0]], dis[n_threshold2[0]]
                      , dis[n_threshold1[0]], dis[n_threshold2[0]]), axis=0)
event_label = np.concatenate((event_label, event_label[n_threshold1[0]], event_label[n_threshold2[0]]
                      , event_label[n_threshold1[0]], event_label[n_threshold2[0]]), axis=0)
#标准化
for i in range(len(data)):
    data[i, :] = data[i, :] * w
for i in range(149):
    data[:, i] = (data[:, i]-data[:, i].mean())/data[:, i].std()


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
    data_batch_loc = index[: int((data_len+1) * (1-test_size))]
    ret = index[int((data_len+1) * (1-test_size)):]
    return data[data_batch_loc], dis[data_batch_loc], label[data_batch_loc], data[ret], dis[ret], label[ret]\
             ,data_batch_loc, ret


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

train = shock_data(x_train, dis_train, y_train)
test = shock_data(x_test, dis_test, y_test)
shock = shock_all(train, test)
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

lam = 0.00002
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

        if i % 100 == 0:
            mse_val = mse.eval(feed_dict={X: shock.train.inputs, Y: shock.train.labels, keep_prob: 1})
            regularizers1 = regularizers.eval(feed_dict={X: shock.train.inputs, Y: shock.train.labels, keep_prob: 1})
            print("step %d/%d, training mse_val %g" % (i, train_size, mse_val))
            print(regularizers1)
            mse_val_test = mse.eval(feed_dict={X: shock.test.inputs, Y: shock.test.labels, keep_prob: 1})
            print("test mse_val %g" % mse_val_test)
            train_l.append(mse_val)
            test_l.append(mse_val_test)

        train_op.run(feed_dict={X: batch[0], Y: batch[2], keep_prob: 0.5})


    # plt.plot(np.array(range(1, 1+len(train_l[1:])))*100, train_l[1:], 's-', c='red', label='Train MSE')
    # plt.plot(np.array(range(1, 1+len(train_l[1:])))*100, test_l[1:], 'o-', c='black', label='CV MSE')
    plt.plot(np.array(range(1, 1 + len(train_l[3:]))) * 100, train_l[3:], c='k', label='Train MSE')
    plt.plot(np.array(range(1, 1 + len(train_l[3:]))) * 100, test_l[3:], c='grey', label='CV MSE')
    np.save("train_l.npy", train_l)
    np.save("test_l.npy", test_l)
    plt.xlabel('STEP', fontsize=26)
    plt.ylabel('MSE', fontsize=26)
    plt.legend()
    plt.savefig('step.png')
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
print(tst.shape, tst[:, 1].shape, p1.shape)
p1 = p1.reshape(len(p1))
df = pd.DataFrame({'event': tst[:, 1], 'logtaoc': p1, 'mag': tst[:, 0]})

# 便于画图，保存数据分割结果
df.to_excel("nn_dataframe.xlsx")


df['event'] = df['event'].apply(lambda x: x[:-1] if len(x)==17 else x)
df['event'] = df['event'].apply(lambda x: x[-10:])
df['mag'] = df['mag'].apply(lambda x: np.float(x))
choose_event_bool = df.groupby(['event']).size() > 4
gb_df = df.groupby(['event']).size().reindex()
choose_event = gb_df.index[choose_event_bool]
df = df[df['event'].isin(choose_event)]

y = df.groupby(['event'])['logtaoc'].mean()
std = df.groupby(['event'])['logtaoc'].std()
plt.hist(np.array(std), bins=20, normed=True, color='k')
plt.xlabel('Variance', fontsize=26)
plt.title("Single event estimation magnitude variance distribution",  fontsize=26)
plt.show()
x = df.groupby(['event'])['mag'].mean()
rmse = np.sqrt(((x-y)*(x-y)).mean())
# c = []
# for i in range(len(std)):
#     c.append([y[i]-std[i], y[i]+std[i]])
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='grey', label='Event Magnitude')
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Prediction $M_w$', fontsize=26)
ax.text(5.3, 3.3, r"$RMSE = %.2f$" % rmse, fontsize=26)
plt.legend()
plt.savefig('nn_full.png')
plt.show()
print(rmse)


df = df[df['mag'] >= 4]
y = df.groupby(['event'])['logtaoc'].mean()
std = df.groupby(['event'])['logtaoc'].std()
plt.hist(np.array(std), bins=20, normed=True, color='k')
plt.xlabel('Variance', fontsize=26)
plt.title("Single event estimation magnitude variance distribution",  fontsize=26)
plt.show()
x = df.groupby(['event'])['mag'].mean()
rmse = np.sqrt(((x-y)*(x-y)).mean())
# c = []
# for i in range(len(std)):
#     c.append([y[i]-std[i], y[i]+std[i]])
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='grey', label='Event Magnitude')
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Prediction $M_w$', fontsize=26)
ax.text(4, 4, r"$RMSE = %.2f$" % rmse, fontsize=26)
plt.legend()
plt.savefig('nn_kana.png')
plt.show()
print(rmse)
