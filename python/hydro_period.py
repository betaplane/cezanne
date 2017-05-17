#!/usr/bin/env python
import pandas as pd
import numpy as np
from functools import partial
import helpers as hh

D = pd.HDFStore('../../data/DGA/DGA.h5')
c = D['caud']
p = D['ppt']
D.close()

q = pd.read_csv('../../data/DGA/Choapa en Salamanca.csv',
                header=None, skiprows=1, index_col=0, parse_dates=True)
r = pd.read_csv('../../data/DGA/-precipitacion-en-san-agustin-[dga].csv',
                header=None, skiprows=1, index_col=0, parse_dates=True)
t = pd.read_csv('../../data/DGA/-temperatura-media-en-la-serena-[ghcn].csv',
                header=None, skiprows=1, index_col=0, parse_dates=True)


x = pd.concat((r, q), 1)
x.columns = ['r', 'q']

def midx(df):
    df.index = pd.DatetimeIndex(['{}-{}'.format(*t) for t in df.index])
    return df
xm = midx(x.groupby((x.index.year, x.index.month)).mean())

def months(start, n, t):
    """
    Groupby function for time axis: group starts at month 'start' and contains 'n' consecutive months.
    The year of the start month is returned as group label; months which fall outside the interval
    are labeled with 0.
    """
    return (t.year - (t.month < start)) * ((t.month - start) % 12 < n)

xg = x.groupby(partial(months, 5, 12)).mean()

b = hh.lsqdf(xg)

xr = xg.q - b['b0'] - b['b1'] * xg.r

def coherence(df, f=None, norm='psd'):
    def t(df):
        df = df.dropna()
        return np.array(df.index, dtype='datetime64[D]').astype(int), df
    if f is None:
        f, X = LombScargle(*t(df.iloc[:,0])).autopower(normalization=norm)
    else:
        X = LombScargle(*t(df.iloc[:,0])).power(f, normalization=norm)
    Y = LombScargle(*t(df.iloc[:,1])).power(f, normalization=norm)
    return f,X,Y# np.abs(X * np.conj(Y))**2 / (np.abs(X)**2 * np.abs(Y)**2)
