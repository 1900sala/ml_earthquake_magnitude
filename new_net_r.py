import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import  pylab
import os



data=np.load('data.npy')
label=np.load('magn.npy')
dis=np.load('dis.npy')
time_window=2000
train_size=25000
trainortest=True
# trainortest=False


# dis=np.array([ [x] for x in dis])
# label=np.array([ [x] for x in label])

print(dis)

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
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1,1,2,1], padding='SAME')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_train,dis_train,y_train,x_test,dis_test,y_test=train_test_split(data,dis,label,test_size=0.2)
# print(len(x_train),len(dis_train),len(y_train),len(x_test),len(dis_test),len(y_test))
sess = tf.InteractiveSession()
train = shock_data(x_train, dis_train,y_train)
test  = shock_data(x_test, dis_test,y_test)
shock=shock_all(train,test)




x = tf.placeholder("float", shape=[None, 1,1000,3])
d = tf.placeholder("float", shape=[None,4])
y_ = tf.placeholder("float", shape=[None])
W_conv1 = weight_variable([1, 25, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([1, 25, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2.shape)


W_conv3 = weight_variable([1, 25, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_conv3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([1, 25, 32, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_conv4 = max_pool_2x2(h_conv4)
print(h_conv4.shape)

W_conv5 = weight_variable([1, 25, 32, 32])
b_conv5 = bias_variable([32])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
h_conv5 = max_pool_2x2(h_conv5)
print(h_conv5.shape)


h_flat1 = tf.reshape(h_conv5, [-1, 4*32])
W_fc1 = weight_variable([4*32, 8])
b_fc1 = bias_variable([8])
h_fc1 = tf.nn.relu(tf.matmul(h_flat1, W_fc1) + b_fc1)
print(h_fc1.shape)
print(d.shape)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.shape)
h_fc1_drop = tf.concat([h_fc1_drop,d],1)
print(h_fc1_drop.shape)
W_fc2 = weight_variable([8+4, 8])
b_fc2 = bias_variable([8])
h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print(h_fc2.shape)

W_fc3 = weight_variable([8, 1])
b_fc3 = bias_variable([1])
y_conv= tf.matmul(h_fc2, W_fc3) + b_fc3

print(y_conv.shape)


regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4) +
                tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3))


mse = tf.reduce_mean(tf.square(y_-y_conv))
loss=mse+0.0001*regularizers
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if trainortest:
    for i in range(train_size):
      batch = shock.train.batch(50)
      # print('batch[0]:',batch[0].shape)
      # print('batch[1]:', batch[1].shape)
      if i%200 == 0:
          mse_val = mse.eval(feed_dict={x:batch[0], d:batch[1],y_: batch[2], keep_prob: 1.0})
          print( "step %d, training mse_val %g"%(i, mse_val))
          regularizers_val = regularizers.eval(feed_dict={x: batch[0], d: batch[1], y_: batch[2], keep_prob: 1.0})
          print("step %d, training regularizers_val %g" % (i, 0.0001*regularizers_val))
          # print( "step %d, training %g"%(i, train_))
      train_step.run(feed_dict={x: batch[0], d:batch[1], y_: batch[2], keep_prob: 0.5})
      if i ==train_size-1 :
          if os.path.exists('./new_net_model_r'):
              saver.save(sess, './new_net_model_r/model.ckpt')
          else:
              os.mkdir('./new_net_model_r')
              saver.save(sess, './new_net_model_r/model.ckpt')
    print("test accuracy %g" % mse.eval(feed_dict={x: shock.test.inputs, d:shock.test.dis,y_: shock.test.labels, keep_prob: 1.0}))
else:
    saver.restore(sess, './new_net_model_r/model.ckpt')


if not trainortest:
    p=y_conv.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    y=shock.test.labels
    m=mse.eval(feed_dict={x: shock.test.inputs, d: shock.test.dis, y_: shock.test.labels, keep_prob: 1.0})
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
    print(m)


    p=y_conv.eval(feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0})
    y=shock.train.labels
    m = mse.eval(feed_dict={x: shock.train.inputs, d: shock.train.dis, y_: shock.train.labels, keep_prob: 1.0})
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(y, c='red',label='y')
    plt.plot(p, c='black',label='prediction_y')
    plt.ylabel(' magnitude')
    plt.legend()
    plt.show()
    print(m)

