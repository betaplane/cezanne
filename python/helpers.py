#!/usr/bin/env python
from datetime import tzinfo, timedelta
from dateutil.parser import parse
import numpy as np
import pandas as pd
import os, re

K = 273.15


def try_list(obj, *args):
    for a in args:
        try:
            return a(obj)
        except Exception as e:
            print(e)


def g2d(v):
    m, n = v.shape[-2:]
    return np.array(v[:]).flatten()[-m * n:].reshape((m, n))


class CEAZAMetTZ(tzinfo):
    def __init__(self):
        self.__offset = timedelta(hours=-4)
        self.__name = self.__class__.__name__

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return timedelta(0)


def get_time(nc):
    from dateutil.parser import parse
    from functools import partial as p
    import re

    def XTIME(var, nc):
        t = nc.variables[var]
        d = {'mi': (14, 'm'), 'ho': (12, 'h')}[t.units[:2]]
        start = np.datetime64(
            parse(t.units[d[0]:]), dtype='datetime64[{}]'.format(d[1]))
        return t[:].astype('timedelta64[{}]'.format(d[1])) + start

    def string(nc):
        t = [''.join(d) for d in nc.variables['Times'][:].astype(str)]
        return np.array(
            ['{}T{}'.format(d[:10], d[11:13]) for d in t], dtype='datetime64')

    def floatdays(var, nc):
        t = nc.variables[var][:]
        tf = np.floor(t).astype(int)
        return np.array(['{}-{}-{}'.format(d[:4],d[4:6],d[6:8]) for d in tf.astype(str)]).astype('datetime64[D]') \
         + np.round((t-tf)*24).astype('timedelta64[h]')

    return try_list(nc,
                    p(XTIME, 'XTIME'),
                    p(XTIME, 'time'), string,
                    p(floatdays, 'Times'), p(floatdays,
                                             'time')).astype('datetime64[h]')


def lonlat(nc):
    return try_list(
        nc, lambda x: (g2d(x.variables['XLONG']), g2d(x.variables['XLAT'])),
        lambda x: (g2d(x.variables['XLONG_M']), g2d(x.variables['XLAT_M'])),
        lambda x: np.meshgrid(x.variables['lon'], x.variables['lat']),
        lambda x: (x['lon'].as_matrix(), x['lat'].as_matrix()))


def ungrib(grb):
    # Note: doesn't give the right time for all grib files, e.g. not the UPP ones
    t, d = zip(*np.array([(m.validDate, m.data()[0]) for m in grb]))
    t = np.array(t, dtype='datetime64[h]')
    d = np.array(d)
    lat, lon = grb[1].latlons()
    return d, t, lon, lat


def glue(name, df, sta=None):
    d = df if sta is None else pd.concat((sta, df), axis=1)
    with pd.HDFStore(os.path.join('data', 'glue', name)) as store:
        store[name] = d


def pyr(df):
    from rpy2 import robjects as ro
    ro.pandas2ri.activate()
    dput = ro.r['dput']
    dput(df, 'py2r.txt')
    ro.pandas2ri.deactivate()


def extract(d, aggr, C2K=False):
    v = d.xs(aggr, level='aggr', axis=1)
    try:
        v = v.drop(v.xs(10, level='elev', axis=1), axis=1)
    except:
        pass
    v.columns = v.columns.get_level_values('station')
    return v + K if C2K else v


def panel_append(P, n):
    d = dict(P)
    d.update(n)
    return pd.Panel(d)


def tsplit(a):
    try:
        return np.where(np.diff(a.index).astype(int) <= 0)[0][0] + 1
    except:
        return np.where(np.diff(get_time(a)).astype(int) <= 0)[0][0] + 1


def day(df):
    d = df.groupby(df.index.date).mean()
    d.index = pd.PeriodIndex(d.index, freq='D')
    return d


def basemap():
    from mpl_toolkits.basemap import Basemap
    return Basemap(
        lat_0=-30.5,
        lat_1=-10.0,
        lat_2=-40.0,
        lon_0=-71.0,
        projection='lcc',
        width=300000,
        height=400000,
        resolution='h')


def lsqdf(*df, b0=True):
    """
least squares regression: predictor in first column/argument, response in second
returns [intercept (b0), slope (b1)]
if b0=False, assumes zero intercept and returns [slope]
	"""
    df = pd.concat(df, axis=1, join='inner')
    X = df.dropna().as_matrix()
    if len(X):
        return lsq(X[:, -1], X[:, :-1])
    else:
        print("no overlap")
        return np.nan


def lsq(Y, X, b0=True):
    """
Y is the label vector, X is a 2D matrix with features in columns. If b0=False, the constant
term is zero.
	"""
    if b0:
        b = np.linalg.lstsq(np.r_['1,2', np.ones((X.shape[0], 1)), X], Y)[0]
    else:
        b = np.linalg.lstsq(X, Y)[0]
    return dict(
        zip(['b{}'.format(i) for i in np.arange(b.shape[0]) + (1 - b0)], b))


def avg(df, interval):
    if interval=='month':
        m = df.groupby((df.index.year,df.index.month)).mean()
        m.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in m.index.tolist()])
    if interval=='hour':
        m = df.groupby((df.index.date,df.index.hour)).mean()
        m.index = pd.DatetimeIndex(['{}T{}:00'.format(*i) for i in m.index.tolist()])
    return m


def stationize(df):
    """
    Return a copy of a DataFrame with only station codes as labels (either columns or index).
    Careful with multiple columns for same station.
    """
    c = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        c.columns = df.columns.get_level_values('station')
    elif isinstance(df.index, pd.MultiIndex):
        c.index = df.index.get_level_values('station')
    return c

def coord_names(xr, *names):
    return [[c for c in xr.coords if re.search(n, c, re.IGNORECASE)][0] for n in names]


def availability_matrix(df, ax=None, label=True, color={}, bottom=.05, top=.99, **kwargs):
    """Plot a matrix of the times when a given :class:`~pandas.DataFrame` has valid observations. Not sure with what data types it'll still work, but in general 0/False(/nan?) should work for nonexistent times, and 1/count for exisitng ones.

    :param df: DataFrame with time in index and station labels as columns. The columns labels are used to label the rows of the plotted matrix.
    :type df: :class:`~pandas.DataFrame`
    :param ax: :obj:`~matplotlib.axes.Axes.axes` if subplots are used
    :param label: if `False`, plot no row labels
    :type label: :obj:`bool`
    :param color: mapping from color values to row indexes whose labels should be printed in the given color
    :type color: :obj:`dict` {color spec: [row indexes]}
    :returns: 
    :rtype: 

    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(kwargs.pop('fig_width', 6), 10 * df.shape[1]/80))
    else:
        fig = ax.figure
    fig.subplots_adjust(bottom=bottom, top=top, **kwargs)
    plt.set_cmap('viridis')
    y = np.arange(df.shape[1] + 1)
    ax.pcolormesh(df.index, y, df.T)
    ax.set_yticks(y[1:])
    if label:
        l = ax.set_yticklabels(df.columns)
        for k in l:
            k.set_verticalalignment('bottom')
            k.set_fontsize(8)
        for c, i in color.items():
            for j in i:
                l[j].set_color(c)
    else:
        ax.set_yticklabels([])
    ax.yaxis.set_tick_params(tick1On=False)
    ax.grid()
    ax.invert_yaxis()
