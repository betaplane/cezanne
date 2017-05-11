#!/usr/bin/env python
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from mapping import basemap
from scipy.interpolate import LinearNDInterpolator, interp1d
from functools import partial
import helpers as hh
from shapely.geometry import Polygon, Point
from scipolate import interp4D
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.figure import SubplotParams

S = pd.HDFStore('../data/IGRA/IGRAraw.h5')
sta = S['sta']


def ts(x):
    # need to do it this way, since dropna above doesn't delete the index properly
    # https://github.com/pandas-dev/pandas/issues/2770
    return np.array(x.unstack().index, dtype='datetime64[h]')



nc = Dataset('../data/wrf/d02_2014-09-10_transf.nc')
ma = basemap(nc)
x, y = ma.xy()
ij = ma(*hh.lonlat(sta.iloc[:2]))
t = hh.get_time(nc)
T = nc.variables['temp'][:]
P = nc.variables['press'][:]

Ti = interp4D((x, y), T, ij, sta.iloc[:2].index, t, method='linear')
Pi = interp4D((x, y), P, ij, sta.iloc[:2].index, t, method='linear')

p = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]
pl = np.log(p) + np.log(100)


def ints(pl, x):
    try:
        return interp1d(
            np.log(x.index), x, 'linear', bounds_error=False)(pl) * .1 + hh.K
    except:
        return np.repeat(np.nan, len(pl))


def intm(P, T, j):
    return interp1d(
        np.log(P.loc[j]), T.loc[j], 'linear', bounds_error=False)(pl)


M = pd.HDFStore('../data/IGRA/IGRAmly.h5')
mly = M['temp']

d = {}
for s in ['ARM00087418', 'CIM00085586']:
    st = S[s]['TEMP'].replace({-9999: np.nan, -8888: np.nan}).dropna()
    m = mly[s].mean().mean(1)['value'].drop(9999) * .1 + hh.K

    i = ts(st)
    k = list(set(i).intersection(t))
    intp = partial(ints, pl)
    tst = pd.DataFrame.from_dict(
        dict([(j, intp(st[j])) for j in i]), orient='index')
    tst.columns = p

    intp = partial(intm, Pi[s], Ti[s])
    tm = pd.DataFrame.from_dict(
        dict([(j, intp(j)) for j in t]), orient='index')
    tm.columns = p

    d[s] = {'mod': tm, 'obs': tst, 'i': k, 'mly': m}


def plot(d, ax):
    dt = d['mod'].loc[d['i']] - d['obs'].loc[d['i']]
    (lambda y: ax.scatter(y, y.index.get_level_values(1)))(dt.stack())
    (lambda y: ax.plot(y, y.index, '-r', label='raw'))(dt.mean())
    (lambda y: ax.plot(y, y.index, '-g', label='monthly')
     )(d['mod'].mean() - d['mly'])
    ax.invert_yaxis()
    ax.grid(which='minor')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim((1050, 90))


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot(d['CIM00085586'], ax[0])
plot(d['ARM00087418'], ax[1])
ylim = np.array([a.get_ylim() for a in ax])
for a in ax:
    a.set_ylim((ylim.max(), ylim.min()))
ax[0].set_title('St Domingo')
ax[1].set_title('Mendoza')
ax[1].set_yticklabels([])

d = {}
for s in ['ARM00087418', 'CIM00085586']:
    st = S[s]['TEMP'].replace({-9999: np.nan, -8888: np.nan}).dropna()
    k = list(set(ts(st)).intersection(t))
    a = np.array([
        list(
            zip(Pi[s].loc[j], Ti[s].loc[j] - ints(np.log(Pi[s].loc[j]), st[
                j]))) for j in k
    ]).flatten().reshape((-1, 2))
    d[s] = a


def id(x):
    x.index = x.index.date
    return x


sm = cm.ScalarMappable(norm=colors.Normalize(vmin=p[-1], vmax=p[0]))
sm.set_array(p)
sm.set_cmap('gnuplot_r')
fig, ax = plt.subplots(2, 2, subplotpars=SubplotParams(left=0.10, right=0.86))
for i, s in enumerate(['CIM00085586', 'ARM00087418']):
    t = d[s]['obs']
    dt = id(t[t.index.hour == 12]) - id(t[t.index.hour == 00])
    x = np.linspace(-10, 10, 100)

    for j in p:
        try:
            g = gaussian_kde(dt[j].dropna())
        except:
            pass
        else:
            ax[0, i].plot(x, g(x), color=sm.to_rgba(j))
    ax[0, i].grid()
    ax[0, i].set_xticklabels([])

    dt = d[s]['mod'] - d[s]['obs']
    for j in p:
        try:
            g = gaussian_kde(dt[j].dropna())
        except:
            pass
        else:
            ax[1, i].plot(x, g(x), color=sm.to_rgba(j))
    ax[1, i].grid()

b1 = ax[0, 1].get_position()
b2 = ax[1, 1].get_position()
cb = plt.colorbar(
    sm, cax=fig.add_axes([b1.x1 + 0.04, b2.y0, 0.02, b1.y1 - b2.y0]))
cb.ax.invert_yaxis()
