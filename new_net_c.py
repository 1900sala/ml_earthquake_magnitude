import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import  pylab
import os
from sklearn.preprocessing import OneHotEncoder


data=np.load('data.npy')
label=np.load('magn.npy')
dis=np.load('dis.npy')

train_size=500000
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
        a=self.inputs[data_batch_loc]
        b=self.dis[data_batch_loc]
        c=self.labels[data_batch_loc]
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
#


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                        strides=[1, 1, 3, 1], padding='SAME')

def preprocessing_magn(magn,f=1/4):
    temp=[]
    min = magn.min()
    # print(min)
    for i in range(len(magn)):
        temp.append([int(round((magn[i]-min)/f))])
    return np.array(temp),min

def return_magn(magn,min,f=1/4):
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




x = tf.placeholder("float", shape=[None, 1,1000,3])
d = tf.placeholder("float", shape=[None,4])
y_ = tf.placeholder("float", shape=[None,magn_c_nums])
W_conv1 = weight_variable([1, 8, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_conv1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([1, 8, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = max_pool_2x2(h_conv2)
print(h_conv2.shape)


W_conv3 = weight_variable([1, 8, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_conv3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([1, 8, 32, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_conv4 = max_pool_2x2(h_conv4)
print(h_conv4.shape)



h_flat1 = tf.reshape(h_conv4, [-1, 13*32])
print(h_flat1.shape)

W_fc1 = weight_variable([13*32, 32])
b_fc1 = bias_variable([32])
h_fc1 = tf.nn.relu(tf.matmul(h_flat1, W_fc1) + b_fc1)
print(h_fc1.shape)

keep_prob = tf.placeholder("float")

W_fc2 = weight_variable([32, 32])
b_fc2 = bias_variable([32])
h_fc2=tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
print(h_fc2.shape)

h_fc2 = tf.concat([h_fc2,d],1)

W_fc3 = weight_variable([32+4, 15])
b_fc3 = bias_variable([15])
h_fc3= tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

h_fc3=tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([15, magn_c_nums])
b_fc4 = bias_variable([magn_c_nums])
y_conv= tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
print(y_conv.shape)


regularizers = k*((tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4) +
                 tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3)+ tf.nn.l2_loss(W_fc4)))

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y_conv,1e-5,1)))
loss=cross_entropy+regularizers
mse = tf.reduce_mean(tf.square(y_-y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if trainortest:
    train_l=[]
    test_l=[]
    mse_l=[]
    for i in range(train_size):
      batch = shock.train.batch(50)

      if i%1000 == 0:
          cross_entropy_val = cross_entropy.eval(feed_dict={x:batch[0], d:batch[1],y_: batch[2], keep_prob: 1.0})
          print("step %d/%d, training cross_entropy_val %g"%(i, train_size, cross_entropy_val))
          regularizers_val = regularizers.eval(feed_dict={x: batch[0], d: batch[1], y_: batch[2], keep_prob: 1.0})
          print("step %d/%d, training regularizers_val %g" % (i, train_size, regularizers_val))
          acc_val = accuracy.eval(feed_dict={x: batch[0], d: batch[1], y_: batch[2], keep_prob: 1.0})
          print("step %d/%d, training acc_val %g" % (i, train_size,acc_val))
          print("test acc_val %g" % accuracy.eval(
              feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))
          print("test mse_val %g" % mse.eval(
              feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))
          train_l.append(acc_val)
          test_l.append(accuracy.eval(
              feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0}))

      train_step.run(feed_dict={x: batch[0], d:batch[1], y_: batch[2], keep_prob: 0.5})
      if i ==train_size-1 :
          if os.path.exists('./new_net_model'):
              saver.save(sess, './new_net_model/model.ckpt')
          else:
              os.mkdir('./new_net_model')
              saver.save(sess, './new_net_model/model.ckpt')


    plt.plot(train_l, c='red', label='train')
    plt.plot(test_l, c='black', label='test')
    plt.ylabel(' acc')
    plt.legend()
    plt.show()
    p=y_conv.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    p=return_magn(p,min)
    y=shock.test.labels
    y=return_magn(y,min)
    m=accuracy.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    print("test acc_val %g" % accuracy.eval(
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
    p=return_magn(p,min)
    y=shock.test.labels
    y=return_magn(y,min)
    m=accuracy.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    print("test acc_val %g" % accuracy.eval(
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
    p=return_magn(p,min)
    y = return_magn(y,min)
    m = accuracy.eval(feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0})
    print("train acc_val %g" % accuracy.eval(
        feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0}))
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
    print(m)
    print(p)
