#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.linalg import solve
import helpers as hh

D = pd.HDFStore('../../data/tables/station_data_new.h5')
S = pd.HDFStore('../../data/tables/LinearLinear.h5')
T = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, 'elev')) + 273.15
Tm = S['T2']
sta = D['sta']
dist = D['dist']
dz = S['z']['d03_op'] - sta['elev']
dt = Tm['d03_0_00']-T
D.close()
S.close()

X = dz.dropna().to_frame()
X[1] = 1
Y = dt[dz.index]
w = dist.loc[dz.index, dz.index]
w[w>50000] = np.nan
w = np.exp(-(w / 20000) ** 2)

def space(v, data, weights=True):
    try:
        y = data.mul(v, 0) if weights else data.loc[v.index[v.notnull()]]
        y.dropna(inplace=True)
    except:
        return np.nan
    if len(y) < 5:
        return np.nan
    x = y.as_matrix()
    b = np.linalg.lstsq(x[:,:2], x[:,2])[0]
    return pd.concat((pd.Series(b*[1000,1], index=['b','c']), data.ix[y.index, 2]))

def time(t, weights=True):
    m = pd.concat((X, t), 1).dropna()
    return w.apply(space, data=m, weights=weights).loc['b']

def mspace(v, data, weights=False):
    try:
        y = data.mul(v, 0, level=0) if weights else data.loc[v.index[v.notnull()].tolist()]
        y = y.dropna()
    except:
        return np.nan
    if len(y) < 5:
        return np.nan
    x = y[[1,'x','y']].as_matrix()
    b = np.linalg.lstsq(x[:,:2], x[:,2])[0]
    return b[1]

def mtime(t, weights=False):
    # from IPython.core.debugger import Tracer; Tracer()()
    x, y = X[0].align(t.T, broadcast_axis=1)
    f = pd.Panel({'x':x, 'y':y}).to_frame()
    f[1] = 1
    return w.apply(mspace, data=f, weights=weights) * 1000

def grouper(t, s=0, h=3):
    return (np.array(t, dtype='datetime64[h]').astype(int) - s) // 3

b = time(Y.loc['2015-08-12 08'])

def plot(c):
    fig = plt.figure()
    y = pd.concat((dz, c, d[c.name]),1)
    plt.scatter(y.iloc[:,0], y.iloc[:,1],c=y.iloc[:,2])
    plt.colorbar()
    plt.scatter(y.ix[c.name,0], y.ix[c.name,1], color='r')
    xl = plt.gca().get_xlim()
    x = np.linspace(xl[0], xl[1], 100)
    plt.plot(x, c['c'] + x * c['b']/1000, '-')
    fig.show()


class GLR(object):
    def __init__(self, dz, dist, s=20000, c=50000):
        self._dz = dz.dropna()
        i = self._dz.index
        w = dist.loc[i, i]
        j = pd.MultiIndex.from_product((i, [0, 1]))
        self._i = dict([(k, pd.MultiIndex.from_product((i, [k]))) for k in [0, 1]])
        w[w > c] = np.nan
        self._w = np.exp(-(w / s) ** 2).fillna(0)
        v = w.notnull()
        self._L = pd.DataFrame(
            np.kron(np.diag(self._w.sum(1)) - self._w, np.identity(2)),
            index = j, columns = j)
        self._C = pd.DataFrame({0:1, 1:dz})
        self._dz1 = self._diag(np.ones(len(i)), 0, 0) + \
            self._diag(self._dz, 1, 0) + \
            self._diag(self._dz, 0, 1)
        self._dz2 = self._diag(self._dz**2, 1, 1)

    def regress(self, X, lda):
        C = self._C.mul(X.sum(), 0).stack()
        i = C.index
        A = self._dz1 * len(X) + self._dz2
        p = solve(A.loc[i, i] + lda * self._L.loc[i, i], C)
        q = pd.DataFrame(p, index=i).unstack()
        q.columns = ['icpt', 'lr']
        return q

    def _diag(self, ii, row, col):
        a = pd.DataFrame(np.diag(ii), index = self._i[row], columns = self._i[col])
        z = np.zeros(a.shape)
        b = pd.DataFrame(z, index = self._i[row], columns = self._i[not col])
        c = pd.DataFrame(z, index = self._i[not row], columns = self._i[col])
        d = pd.DataFrame(z, index = self._i[not row], columns = self._i[not col])
        return pd.concat((
            pd.concat((a, b), 1).sort_index(1),
            pd.concat((c, d), 1).sort_index(1)
        ), 0).sort_index(0)

    def plot(self, X, p, loc):
        fig = plt.figure()
        w = self._w[loc]
        w = w[w>0]
        i = w.index
        x = X[i].as_matrix().T.flatten()
        plt.scatter(np.repeat(self._dz[i], len(X)), x, c=np.repeat(w, len(X)))
        plt.colorbar()
        plt.scatter(np.repeat(self._dz[loc], len(X)), X[loc], color='r')
        xl = plt.gca().get_xlim()
        y = np.linspace(xl[0], xl[1], 100)
        plt.plot(y, c['c'] + x * c['b']/1000, '-')
        fig.show()
