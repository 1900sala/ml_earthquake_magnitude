import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import  pylab
import os
from sklearn.preprocessing import OneHotEncoder


data=np.load('data_filt.npy')
label=np.load('magn.npy')
dis=np.load('dis.npy')

train_size=10000
trainortest=True
# trainortest=False
k=0.001





def train_test_split(data,dis,label,test_size=0.3):
    data_len=len(data)
    nums=int(data_len*(1-test_size))
    data_batch_loc = np.random.randint(data_len, size=nums)
    all_loc=range(data_len)
    ret = [i for i in all_loc if i not in data_batch_loc]
    return  data[data_batch_loc],dis[data_batch_loc],label[data_batch_loc],data[ret],dis[ret],label[ret]


class shock_data(object):
    def __init__(self, inputs,dis, labels):
        self.inputs = inputs
        self.dis = dis
        self.labels = labels
    def batch(self,size):
        data_len=len(self.inputs)
        data_batch_loc=np.random.randint(data_len, size=size)
        a = self.inputs[data_batch_loc]
        b = self.dis[data_batch_loc]
        c = self.labels[data_batch_loc]
        # print(np.shape(b))
        # print(b)
        return a,b,c
class shock_all(object):
    def __init__(self, train,test):
        self.train = train
        self.test = test

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_with_weight_decay( shape, stddev, wd):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                            initializer=initializer)
    return var

def conv_layer( bottom, name, shape=None):
    with tf.variable_scope(name) as scope:
        stddev = 0.05
        initW = tf.truncated_normal_initializer(stddev=stddev)
        filter = tf.get_variable(name='filter', shape=shape, initializer=initW)
        initB = tf.constant_initializer(0.0)
        conv_bias = tf.get_variable(name='bias', shape=shape[3], initializer=initB)
        conv = tf.nn.conv2d(bottom, filter, strides=[1, 2, 2, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
    return relu, filter

def fc_layer(bottom, name, shape=None):
    with tf.variable_scope(name) as scope:
        stddev = 0.05
        wd = 5e-4
        weight = variable_with_weight_decay(shape, stddev, wd)
        initB = tf.constant_initializer(0.0)
        bias = tf.get_variable(name='bias', shape=shape[1], initializer=initB)
        fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
        if name == 'output':
            return fc ,weight
        else:
            relu = tf.nn.relu(fc)
    return relu, weight



def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                        strides=[1, 1, 3, 1], padding='SAME')


def preprocessing_magn(magn,f=0.3):
    temp=[]
    min = magn.min()
    # print(min)
    for i in range(len(magn)):
        temp.append([int(round((magn[i]-min)/f))])
    return np.array(temp),min

def return_magn(magn,min,f=0.3):
    temp = []
    for i in range(len(magn)):
        temp.append(np.argmax(magn[i])*f+min)
    return np.array(temp)

label,min=preprocessing_magn(label)
magn_c_nums=label.max()+1
enc = OneHotEncoder()
enc.fit(label)
label=enc.transform(label).toarray()

x_train,dis_train,y_train,x_test,dis_test,y_test=train_test_split(data,dis,label,test_size=0.2)
sess = tf.InteractiveSession()
train = shock_data(x_train, dis_train,y_train)
test  = shock_data(x_test, dis_test,y_test)
shock=shock_all(train,test)



x = tf.placeholder("float", shape=[None, 1,1200,3])
d = tf.placeholder("float", shape=[None,4])
y_ = tf.placeholder("float", shape=[None])



conv1, W_conv1 = conv_layer(x, 'conv1', [1,4,3,32])
print ('conv1.shape', conv1.shape)
conv2, W_conv2= conv_layer(conv1, 'conv2', [1,4,32,32])
print ('conv2.shape', conv2.shape)
conv3, W_conv3 = conv_layer(conv2, 'conv3', [1,4,32,32])
print ('conv3.shape', conv3.shape)
conv4, W_conv4= conv_layer(conv3, 'conv4', [1,4,32,32])
print ('conv4.shape', conv4.shape)
conv5, W_conv5 = conv_layer(conv4, 'conv5', [1,4,32,32])
print ('conv5.shape', conv5.shape)
conv6, W_conv6 = conv_layer(conv5, 'conv6', [1,4,32,32])
print ('conv6.shape', conv6.shape)
conv7, W_conv7 = conv_layer(conv6, 'conv7', [1,4,32,32])
print ('conv7.shape', conv7.shape)
conv8, W_conv8 = conv_layer(conv7, 'conv8', [1,4,32,32])
print ('conv8.shape', conv8.shape)

conv8_shape = conv8.get_shape().as_list()
h_flat1 = tf.reshape(conv8, [-1, conv8_shape[2]* conv8_shape[3]])
print('h_flat1.shape', h_flat1.shape)

h_fc1, W_fc1 = fc_layer(h_flat1, 'fc1',shape= [conv8_shape[2]* conv8_shape[3],16])
print('h_fc1.shape',h_fc1.shape)

h_fc1 = tf.concat([h_fc1,d],1)
keep_prob = tf.placeholder("float")
h_fc2, W_fc2 = fc_layer(h_fc1, 'fc2',shape= [20,16])
print('h_fc2.shape',h_fc2.shape)

h_fc2=tf.nn.dropout(h_fc2, keep_prob)

y_conv, W_out = fc_layer(h_fc2, 'output',shape= [16,1])
print(y_conv.shape)


regularizers = k*((tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4) +
                 tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_conv6) + tf.nn.l2_loss(W_conv7) +tf.nn.l2_loss(W_conv8) +
                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_out)))



cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y_conv,1e-5,1)))
loss=cross_entropy+regularizers

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if trainortest:
    train_l=[]
    test_l=[]
    mse_l=[]
    for i in range(train_size):
      batch = shock.train.batch(50)

      if i%1000 == 0:
          regularizers_val = regularizers.eval(feed_dict={x:batch[0], d:batch[1],y_: batch[2], keep_prob: 1.0})
          print("step %d/%d, training regularizers_val %g"%(i, train_size, regularizers_val))
          mse_val = mse.eval(feed_dict={x: batch[0], d: batch[1], y_: batch[2], keep_prob: 1.0})
          print("step %d/%d, training mse_val %g" % (i, train_size,mse_val))
          print("test mse_val %g" % mse.eval(
              feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))

          train_l.append(mse_val)
          test_l.append(mse.eval(
              feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))

      train_step.run(feed_dict={x: batch[0], d:batch[1], y_: batch[2], keep_prob: 0.5})
      if i ==train_size-1 :
          if os.path.exists('./new_net_model'):
              saver.save(sess, './new_net_model/model.ckpt')
          else:
              os.mkdir('./new_net_model')
              saver.save(sess, './new_net_model/model.ckpt')


    plt.plot(train_l[4:], c='red', label='train')
    plt.plot(test_l[4:], c='black', label='test')
    plt.ylabel(' acc')
    plt.legend()
    plt.show()
    p=y_conv.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    y=shock.test.labels
    print("test mse_val %g" % mse.eval(
        feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
else:
    saver.restore(sess, './new_net_model/model.ckpt')


if not trainortest:
    p=y_conv.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    y=shock.test.labels
    m=mse.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    print("test mse_val %g" % mse.eval(
        feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
    print(m)
    print(p)


    p=y_conv.eval(feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0})
    y=shock.train.labels
    m = mse.eval(feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0})
    print("train acc_val %g" % mse.eval(
        feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0}))
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
    print(m)
    print(p)

