import numpy as np


def moving_average(x, n, type='simple'):
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a-a.mean()


def find_index(x,threshold):
    for i in range(len(x)):

        if np.abs(x[i])>np.abs(threshold):
            return i
#
# p=read('..\data\AIC0081708251205.EW')
#
# pp=moving_average(p[0].data,10,'simple')
# plt.plot(p[0].data-p[0].data.mean())
# plt.plot(pp-pp.mean())
# plt.show()
