#!/usr/bin/env python
from datetime import tzinfo, timedelta
import numpy as np
import pandas as pd
from traitlets.config.loader import PyFileConfigLoader
from functools import singledispatch
from importlib import import_module
import os, re

# config = PyFileConfigLoader(os.path.expanduser('~/Dropbox/work/config.py')).load_config()

def __get_config():
    import sys
    sys.path.append(os.path.expanduser('~/Dropbox/work'))
    import config_mod
    return config_mod

config = __get_config()
sta = pd.read_hdf(config.CEAZAMet.meta_data, 'stations')
flds = pd.read_hdf(config.CEAZAMet.meta_data, 'fields')

K = 273.15

def try_list(obj, *args):
    for a in args:
        try:
            return a(obj)
        except Exception as e:
            print(e)

def drop_duplicates(arr, dim):
    """Drop duplicates from an :class:`xarray.DataArray` dimension (the first occurence of a duplicate label is retained).

    :param arr: array on which to operate (is not modified)
    :type arr: :class:`xarray.DataArray`
    :param dim: name of dimesion along which to look for duplicates
    :type dim: :obj:`str`
    :returns: new array with duplicates dropped
    :rtype: :class:`xarray.DataArray`

    """
    idx = arr.indexes[dim]
    j = np.hstack(idx.get_indexer_for([i])[1:] for i in idx.get_duplicates())
    return arr.isel(**{dim: list(set(range(len(idx))) - set(j))})

def data_gaps(x, gap_size):
    """Get indexes of those columns of a :class:`pandas.DataFrame` which do not have gaps larger or equal to ``gap_size``.

    :param x: DataFrame to analyse - should be regularly resampled
    :type x: :class:`pandas.DataFrame`
    :param gap_size: minimum break length given as number of samples
    :type gap_size: :obj:`int`
    :returns: indexes of the columns that don't have gaps larger than ``gap_size``
    :rtype: :class:`pandas.Index`

    """
    tree = import_module('sklearn.tree')
    tr = tree.DecisionTreeClassifier(min_samples_leaf = gap_size)
    t = np.array(x.index, dtype='datetime64[m]', ndmin=2).astype(float).T
    y = x.isnull().astype(int).apply(lambda c: tr.fit(t, c).predict(t), 0)
    return y.columns[y.sum() == 0]

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


def avg(df, interval):
    if interval=='month':
        m = df.groupby((df.index.year,df.index.month)).mean()
        m.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in m.index.tolist()])
    if interval=='hour':
        m = df.groupby((df.index.date,df.index.hour)).mean()
        m.index = pd.DatetimeIndex(['{}T{}:00'.format(*i) for i in m.index.tolist()])
    return m

@singledispatch
def stationize(df, aggr='prom'):
    """ Return a copy of a DataFrame with only station codes as column labels. If the resulting set of column lables is not unique (more than one sensor for the same variable at the same station), the returned copy has the ``sensor_code`` as column labels.

    :param df: Input DataFrame with :class:`pandas.MultiIndex` in columns. If :obj:`str`, interpret it as the key to load from the ``.h5`` file specified as :attr:`data.CEAZAMet.station_data`.
    :type df: :class:`~pandas.DataFrame` (or :obj:`str`)
    :param aggr: If the input DataFrame has several ``aggr`` levels (e.g. ``prom``, ``min``, ``max``), return this one.
    :type aggr: :obj:`str`
    :returns: DataFrame with simple column index (containing station labels, or sensor codes in case the station index in not unique).
    :rtype: :class:`~pandas.DataFrame`

    """
    try:
        df = df.xs(aggr, 1, 'aggr')
    except KeyError:
        pass

    stations = df.columns.get_level_values('station')
    if len(stations.get_duplicates()) > 0:
        df.columns = df.columns.get_level_values('sensor_code')
    else:
        df.columns = stations
    return df

@stationize.register(str)
def stationize_str(s, *args, **kwargs):
    df = pd.read_hdf(config.Field.file_name, s)
    return stationize(df, *args, **kwargs)

def coord_names(xr, *names):
    return [[c for c in xr.coords if re.search(n, c, re.IGNORECASE)][0] for n in names]

def table(df):
    """Format a small :class:`~pandas.DataFrame` as an `org-mode table <https://orgmode.org/manual/Tables.html>`_.

    :param df: input DataFrame
    :type df: :class:`~pandas.DataFrame`
    :returns: org-mode table as IPython display string with 'text/org' MIME type

    """
    disp = import_module('IPython.display')

    def index(idx):
        if isinstance(idx, pd.MultiIndex):
            x = list(idx)
            return [x[0]] +[[' ' if x[i][j] == z else z for j, z in enumerate(y)]
                            for i, y in enumerate(x[1:])]
        else:
            return [[i] for i in idx]

    idx = index(df.index)
    cols = index(df.columns)
    M = df.as_matrix()
    s = '|\n|'.join('|'.join(' ' for _ in range(len(idx[0]))) + '|' + \
                          '|'.join(str(c[i]) for c in cols) for i in range(len(cols[0]))) + \
        '|\n|' + '|'.join('-' for _ in range(len(idx[0]) + len(M[0]))) + '|\n|' + \
        '|\n|'.join('|'.join(str(i) for j in z for i in j) for z in zip(idx, M))
    return disp.publish_display_data({'text/org': '|' + s + '|'})
