
import pandas as pd
import lightgbm as lgb
import  sklearn
import numpy as np
import matplotlib.pyplot as plt;plt.rcdefaults()
import matplotlib.pyplot as plt
import random



input_vec_size = lstm_size = 10  # 输入向量的维度
time_step_size = 149  # 循环层长度

train_size = 15000
batch_size = 50
data1 = np.load('n_nn_taoc_data.npy')
data2 = np.load('n_nn_taoc_data57.npy')
data = np.concatenate((data1, data2), axis=0)
label1 = np.load('n_magn.npy')
label2 = np.load('n_magn57.npy')
label = np.concatenate((label1, label2), axis=0)
w = np.load('w.npy')
# label = label.reshape(len(label), 1)
dis1 = np.load('n_dis.npy')
dis2 = np.load('n_dis57.npy')
dis = np.concatenate((dis1, dis2), axis=0)
f_threshold = np.where((label > 3.4)&(label < 5.8))
data = data[f_threshold[0]]
label = label[f_threshold[0]]
dis = dis[f_threshold[0]]
#标准化
for i in range(len(data)):
    data[i, :] = data[i, :]*w
for i in range(149):
    data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()

# df = pd.DataFrame(data)
# df['label'] = label
data_len = len(data)
index = np.array(range(data_len))
random.shuffle(index)
data_batch_loc = index[ : int((data_len+1)*0.7)]
ret = index[int((data_len+1)*0.7) :]
c = [str(i) for i in range(149)]
print('formating for lgb')

pd_train = pd.DataFrame(data=data[data_batch_loc], columns=c)
d_train = lgb.Dataset(pd_train, label[data_batch_loc])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'num_leaves': 96,
    'max_depth': 20,
    # 'feature_fraction': 0.9,
    # 'bagging_fraction': 0.95,
    # 'bagging_freq': 5
}
ROUNDS = 100
bst = lgb.train(params, d_train, ROUNDS)

feature_name=bst.feature_name()
feature_importance=bst.feature_importance()/(bst.feature_importance().sum())
#画图

fi = pd.Series(feature_importance,index = feature_name)
fi.plot(kind = 'barh',color='b',alpha = 0.7)
plt.show()

### build candidates list for test ###

print('light GBM predict')
pd_test = pd.DataFrame(data=data[ret], columns=c)
d_test = lgb.Dataset(pd_test)
preds = bst.predict(pd_test)
print(np.mean(np.abs(preds - label[ret])))
preds = bst.predict(pd_train)
print(np.mean(np.abs(preds - label[data_batch_loc])))
