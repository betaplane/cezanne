#!/usr/bin/env python
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import tree
from collections import Counter
from functools import singledispatch


def time_irreg(df):
    """Check time index of DataFrame for irregularities.

    :param df: pandas.DataFrame with time index
    :returns: DataFrame with columns of irregular time intervals before and/or after a given index;
    list of irregular indexes ocurring in groups; list of 'lonely' irrgular indexes.
    :rtype: pandas.DataFrame, list of numpy.arrays, numpy.array

    """
    t = np.array(df.index, dtype='datetime64[m]')
    dt = np.diff(t).astype(float)
    # counts distinct elements in dt
    c = Counter(dt).most_common(1)[0][0]

    # look for indexes which are != the most common timestep on both sides
    d = np.r_[np.nan, dt, dt, np.nan].reshape((2,-1))
    i = (d != c).all(0) # not ok [ok: (d == c).any(0)]
    # DataFrame with intervals before and after timestamp
    f = pd.DataFrame(d[:,i].T, index=df.index[i])

    # look for groups of not ok indexes
    p = np.where(i)[0]
    q = np.where(np.diff(p) != 1)[0]
    q = np.r_['0', [-1], q, q , [-2]] + 1
    # groups
    g = [p[x[0]:x[1]] for x in q.reshape((2, -1)).T if np.diff(x) > 1]
    # 'lonely' indexes
    l = np.array(sorted(set(p) - set([x for y in g for x in y])))
    return f, g, l


@singledispatch
def bin(df, start_minute=0, freq=60, label='end', min_samples_leaf=1000):
    """Average raw data from CEAZAMet stations (fetched by CEAZAMet.Downloader.fetch_raw) into true time intervals.

    Points about the algorithm:

    1) It is assumed that the logger's timestamp refers to the end of a sampling interval.

    2) A decision tree is used to split the time series into parts with likely same sampling interval. For each found part, the binning is applied separately, since weights according to the sampling interval are applied for the averaging (see point 3).

    3) If a sampling interval straddles the boundary between two binning intervals, the record's average is distributed proportionally between the adjacent binning intervals. The max / min values are binned with that binning interval that covers the larger part of the sampling interval, since it is impossible to tell on which side of the boundary the max / min was recorded.

    :param df: data to be averaged / binned
    :type df: pandas.DataFrame with 'avg', 'min', 'max' labels at level 'aggr' in the columns MultiIndex
    :param start_minute: minute within an hour at which a binning interval starts
    :param freq: length of binning interval (in minutes)
    :param label: at what timepoint to label the result: 'start', 'middle' or 'end' (default) of interval
    :returns: averaged and max / min values of data
    :rtype: pandas.DataFrame with same columns as input

    """
    t = np.array(df.index, dtype='datetime64[m]').astype(int)
    dt = np.diff(t)
    # use min of dt on either side of timestamp as label
    d = np.r_[dt[0], dt, dt, dt[-1]].reshape((2,-1)).min(0)
    T = t.reshape((-1, 1))

    # a decision tree is used to split the time series according to the length of record intervals
    tr = tree.DecisionTreeClassifier(min_samples_leaf = min_samples_leaf)
    tr.fit(T, d.T)
    cl = tr.predict(T)
    s = sorted(set(cl))

    print('{} record intervals detected ({}) minutes'.format(len(s), s))

    # here the actual binning routine is called separately for each 'class' of record intervals
    y = pd.concat([aggregate(df[cl==c], t[cl==c], c, start_minute, freq, label)
                      for c in s], axis=1, keys=s)

    cols = df.columns.tolist()
    x = pd.concat([{
        'avg': lambda x: x.mean(1, level='sensor_code'),
        'min': lambda x: x.min(1, level='sensor_code'),
        'max': lambda x: x.max(1, level='sensor_code'),
    }[a[-1]](y.xs(a[-1], 1, 'aggr')) for a in cols], axis=1, keys = cols)
    x.columns = x.columns.droplevel('sensor_code')
    x.columns.names = df.columns.names
    return x


def aggregate(df, t, c, start_minute, freq, label):
    print('aggregating records with {}-min interval'.format(c))

    # compute binning intervals
    ti = t.astype(int) - c                                  # start point of record intervals
    ts = start_minute + (ti - start_minute) // freq * freq  # same label for intervals of length 'freq' from global origin
    v = ti + c - ts - freq
    k = (v > 0)               # record intervals split by end of binning intervals
    l = (v <= 0)              # records completely within binning intervals
    w = v[k].reshape((-1, 1)) # minutes of record intervals falling into next binning interval
    ts = ts.reshape((-1, 1))

    # for mean, split record intervals proportionally into adjacent binning intervals
    ave = df.xs('avg', 1, 'aggr', False)
    cols = ave.columns
    a = ave.as_matrix()
    aft = np.r_['1', a[k] * w, w, ts[k] + freq]
    bef = np.r_['1', a[k] * (c - w), c - w, ts[k]]
    a = np.r_['1', a[l] * c, np.ones((sum(l), 1)) * c, ts[l]]
    a = np.r_['0', a, bef, aft]
    ave = pd.DataFrame(a).groupby(a.shape[1] - 1).sum()
    ave = ave.iloc[:, :-1].divide(ave.iloc[:, -1], 0)
    ave.columns = cols

    def col(x, c):
        x.columns = pd.MultiIndex.from_tuples(x.columns, names=df.columns.names)
        return x.xs(c, 1, 'aggr', False)

    # for min and max, use the binning interval with the largest overlap with the record interval
    ts[v > c/2] = ts[v > c/2] + freq
    b = df.drop('avg', 1, 'aggr')
    b.columns = b.columns.tolist() # avoid warning due to joining MultiIndex to single index
    g = b.join(pd.DataFrame(ts, index=df.index, columns=['ts'])).groupby('ts')
    D = pd.concat((col(g.min(), 'min'), ave, col(g.max(), 'max')), 1)

    lab = pd.Timedelta({'end': freq, 'middle': freq/2, 'start': 0}[label], 'm')
    D.index = pd.DatetimeIndex(np.array(D.index, dtype='datetime64[m]').astype(datetime)) + lab
    return D.sort_index(1)

# wrapper so that bin() can be called with a pandas.HDFStore as first argument
# Note: this is presumed to work for the case that
# 1) the DataFrames are organized according to station, and
# 2) the datalogger samples all variables at the same times
@bin.register(pd.HDFStore)
def bin_store(d, *args, **kwargs):
    return pd.concat([binning(d[k], *args, **kwargs) for k in d.keys()], axis=1)
