#!/usr/bin/env python
import numpy as np
import pandas as pd
import helpers as hh

D = pd.HDFStore('../../data/tables/station_data_new.h5')
S = pd.HDFStore('../../data/tables/LinearLinear.h5')
T = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, 'elev')) + 273.15
Tm = S['T2']
sta = D['sta']
d = D['dist']
dz = S['z']['d03_op'] - sta['elev']
dt = Tm['d03_0_00']-T

X = dz.dropna().to_frame()
X[1] = 1
Y = dt[dz.index]
w = d.loc[dz.index, dz.index]
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
    from IPython.core.debugger import Tracer; Tracer()()
    x, y = X[0].align(t.T, broadcast_axis=1)
    f = pd.Panel({'x':x, 'y':y}).to_frame()
    f[1] = 1
    return w.apply(mspace, data=f, weights=weights) * 1000

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
