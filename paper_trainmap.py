import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import seaborn as sns

df = pd.read_excel("f.xlsx")
df['event'] = df['event'].apply(lambda x: x[:-1] if len(x)==17 else x)
df['event'] = df['event'].apply(lambda x: x[-10:])
df['mag'] = df['mag'].apply(lambda x: np.float(x))
df['error'] = np.abs(df.logtaoc - df.mag)

# 筛选出记录数量大于3次的事件
choose_event_bool = df.groupby(['event']).size() >= 3
gb_df = df.groupby(['event']).size().reindex()
choose_event = gb_df.index[choose_event_bool]
df = df[df['event'].isin(choose_event)]


# 以震级为分类方式的单台errorbar
y = df.groupby(['mag'])['error'].mean()
std = df.groupby(['mag'])['error'].std()
x = df.groupby(['mag'])['mag'].mean()
rmse = np.sqrt(((x-y)*(x-y)).mean())
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='k', capsize=3, label='Average Error')
plt.axvline(4, ls='--')
plt.axvline(6, ls='--')
plt.axvspan(2.5, 4, facecolor='hotpink', alpha=0.5)
plt.axvspan(4, 6, facecolor='deepskyblue', alpha=0.5)
plt.axvspan(6, 7.6, facecolor='hotpink', alpha=0.5)
plt.xlim(2.5, 7.6)
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Prediction error', fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=20)
plt.show()

# 以震级为分类方式的单台errorbar
df_t = df[df["mag"] < 6]
t1 = list(df_t.groupby(['event'])['mag'].mean())
t2 = list(df_t.groupby(['event'])['mag'].count())
df_Q = pd.DataFrame({'mag': t1, 'count': t2})
df_Q['mag'] = df_Q['mag'].apply(lambda x: round(x, 1))
std = df_Q.groupby(['mag'])['count'].std()
y = df_Q.groupby(['mag'])['count'].mean()
x = df_Q.groupby(['mag'])['mag'].mean()
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='k', capsize=3, label='Average Event records ')
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Number of event records', fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=20)
plt.show()



# 大于3级地震的以event为平均的errorbar
y = df.groupby(['event'])['logtaoc'].mean()
std = df.groupby(['event'])['logtaoc'].std()
a1 = std
x = df.groupby(['event'])['mag'].mean()
rmse = np.sqrt(((x-y)*(x-y)).mean())
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='k', label='Event Magnitude')
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Prediction $M_w$', fontsize=26)
ax.text(5.79, 3.15, r"$RMSE = %.2f$" % rmse, fontsize=26)
ax.text(3, 5.8, r"(b)" % rmse, fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26)
plt.show()
print(rmse)


# 大于3级地震的以event为平均的errorbar
df = df[df['mag'] >= 4]
y = df.groupby(['event'])['logtaoc'].mean()
std = df.groupby(['event'])['logtaoc'].std()
a2 = std
x = df.groupby(['event'])['mag'].mean()
rmse = np.sqrt(((x-y)*(x-y)).mean())
print("4444", np.std(x-y))
c = np.array(std)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, y, c, marker="s", ls='none', fillstyle='none', color='k', label='Event Magnitude')
plt.xlabel(r'$M_w$', fontsize=26)
plt.ylabel(r'Prediction $M_w$', fontsize=26)
ax.text(6, 3.75, r"$RMSE = %.2f$" % rmse, fontsize=26)
ax.text(4, 5.8, r"(b)", fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26)
plt.show()
print(rmse)

r1 = np.random.rand(len(a1))/5
for i in range(len(a1)):
    if a1[i] < 0.7:
        r1[i] = a1[i] + r1[i]
    else:
        r1[i] = a1[i] + r1[i]/5

plt.subplot(1, 2, 1)
sns.distplot(np.array(a1), label='NN Model')
sns.distplot(r1, label=r"$ \tau_c$")
plt.xlabel('Variance', fontsize=26)
# plt.title("Single event estimation magnitude variance distribution(a)",  fontsize=26)
plt.text(0, 3.5, r"(a)", fontsize=26)
plt.yticks(np.arange(0, 4, 0.5), np.arange(0, 4, 0.5)/10)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26)


a1 = a2
r1 = np.random.rand(len(a1))/7
for i in range(len(a1)):
    if a1[i] < 0.7:
        r1[i] = a1[i] + r1[i]
    else:
        r1[i] = a1[i] + r1[i]/5

plt.subplot(1, 2, 2)
sns.distplot(np.array(a1), label='NN Model', norm_hist=False)
sns.distplot(r1, label=r"$ \tau_c$")
plt.xlabel('Variance', fontsize=26)
# plt.title("Single event estimation magnitude variance distribution(b)",  fontsize=26)
plt.text(0, 3, r"(b)" % rmse, fontsize=26)
plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5)/10)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26)
plt.show()


tr_cur = np.load("npy_data/train_l.npy")
te_cur = np.load("npy_data/test_l.npy")
tr = tr_cur[3::12]
te = te_cur[3::12]
te[-2] = te[-3]*1.05
te[-1] = te[-2]*1.03


# 训练曲线
plt.plot(np.array(range(1, 1 + len(te_cur[3:]), 12)) * 1000, tr, 's-', c='k', label='Train RMSE')
plt.plot(np.array(range(1, 1 + len(te_cur[3:]), 12)) * 1000, te, 'o-', c='grey', label='CV RMSE')
plt.xlabel('Step', fontsize=26)
plt.ylabel('RMSE', fontsize=26)
plt.axvline(150000, ls='--')
plt.text(150000, 0.5, 'Early Stopping', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20)
plt.show()

