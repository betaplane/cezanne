#!/usr/bin/env python
import numpy as np
import pandas as pd
import ephem as ep
# import pysolar as ps
from functools import partial
import matplotlib.pyplot as plt
from scipy import interpolate as ip
from mapping import basemap
from netCDF4 import Dataset
import helpers as hh
from scipolate import interp4D
from scipy.stats import binned_statistic



D = pd.HDFStore('../../data/tables/station_data_new.h5')
rs = D['rs_w']
sta = D['sta']

So = 1361
sun = ep.Sun()
rad = np.pi / 180


def hour(obs, t):
    obs.date = t
    sun.compute(obs)
    return obs.sidereal_time() - sun.ra


def dec(obs, t):
    obs.date = t
    sun.compute(obs)
    return sun.dec


# extra-terrestrial radiation
# Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. FAO Irrigation and drainage paper No. 56.
# Rome: Food and Agriculture Organization of the United Nations 56, 97–156.
def extraterrestrial(stations, time):
    t = pd.date_range(time[0], time[-1] + np.timedelta64(1, 'h'), freq='H')
    Ra = pd.DataFrame(index=time)
    for c, r in sta.iterrows():
        print(c)
        obs = ep.Observer()
        obs.lon = r.lon * rad
        obs.lat = r.lat * rad
        obs.elevation = r.elev
        f = partial(hour, obs)
        g = partial(dec, obs)
        te = t + np.timedelta64(4, 'h') - np.timedelta64(int(r.interval), 's')
        w = np.array([f(i) for i in te])
        d = np.array([g(i) for i in te])
        ws = np.arccos(-np.tan(obs.lat) * np.tan(d))
        w[w > np.pi] -= 2 * np.pi
        w[w < -np.pi] += 2 * np.pi
        w[w < -ws] = 0
        w[w > ws] = 0
        w1 = w[:-1]
        w2 = w[1:]
        d = (d[:-1] + d[1:]) / 2
        R = 12 / np.pi * So * (1 + 0.033 * np.cos(2 * np.pi / 365 * t[:-1].dayofyear)) * \
        ((w2 - w1) * np.sin(obs.lat) * np.sin(d) + np.cos(obs.lat) * np.cos(d) * (np.sin(w2) - np.sin(w1)))
        R[R < 0] = 0
        return pd.concat((Ra, pd.Series(R, index=rs.index, name=c)), axis=1)


# Ra = extraterrestrial(sta, rs.index)
Ra = D['Ra_w']


# regression coefficients clear-sky from extra-terrestrial radiation
# 'b' - regression coeff
# 'n' - number of points considered (selecting only times with stronges
def regression(rm, Ra):
    b = []
    for c, r in rm.iteritems():
        x = pd.concat((Ra[c[0]], r), axis=1)
        x.iloc[:, 1][x.iloc[:, 1] > x.iloc[:, 0]] = np.nan
        xd = x.groupby(x.index.date).sum()
        n = 2 * (xd.iloc[:, 1] - 0.25 * xd.iloc[:, 0]) / xd.iloc[:, 0]
        n = n.reindex(x.index, method='ffill')
        y = x[n > .9]

        try:
            for i in range(2):
                X = y.dropna().as_matrix()
                a = np.linalg.lstsq(X[:, :1], X[:, 1:])[0][0][0]
                y = x[x.iloc[:, 1] > Ra[c[0]] * a]
        except:
            b.append((np.nan, 0))
        else:
            b.append((a, y.count().min()))
    return pd.DataFrame(b, index=rm.columns, columns=['b', 'n'])


b = regression(rs.xs('prom', level='aggr', axis=1), Ra)
# stdom -  73 m
# mend  - 705 m

# S = pd.HDFStore('../data/IGRA/IGRAraw.h5')
# stdom = S['stdom'][['TEMP', 'GPH']].replace({
#     -9999: np.nan,
#     -8888: np.nan
# }).dropna(0, 'any')
# mend = S['mend'][['TEMP', 'GPH']].replace({
#     -9999: np.nan,
#     -8888: np.nan
# }).dropna(0, 'any')


# def interp(i, X, z):
#     try:
#         return ip.interp1d(X.loc[i]['GPH'], X.loc[i]['TEMP'], 'linear')(z)
#     except:
#         return np.nan


# t50st = pd.DataFrame.from_dict(
#     dict([(i, interp(i, stdom, 123)) for i in m.index.levels[0]]),
#     orient='index')
# t50m = pd.DataFrame.from_dict(
#     dict([(i, interp(i, mend, 755)) for i in m.index.levels[0]]),
#     orient='index')


nc = Dataset('../../data/WRF/2d/d02_2014-09-10_transf.nc')
ma = basemap(nc)
x, y = ma.xy()
ij = ma(*hh.lonlat(sta))
t = hh.get_time(nc)
T = nc.variables['temp'][:]
GP = nc.variables['ghgt'][:]
HGT = nc.variables['HGT'][:]

Ti = interp4D((x, y), T, ij, sta.index, t, method='linear')
Gi = interp4D((x, y), GP, ij, sta.index, t, method='linear')


Tm = np.mean(T,0)
Gm = np.mean(GP,0)

binned_statistic(Gm.flatten(), Tm.flatten(), 'std', 50)

d = {}
for s,r in Ti.iteritems():
    z = sta['elev'][s] + 50
    itpl = []
    for t,c in r.iterrows():
        itpl.append(ip.interp1d(Gi[s].loc[t], c, 'linear', bounds_error=False)(z))
    d[s] = np.mean(itpl)
