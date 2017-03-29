#!/usr/bin/env python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


S = pd.HDFStore('../../data/hydro/data.h5')
D = S['data']
el = D['Elqui']
li = D['Limari']

def midx(df):
    df.index = pd.PeriodIndex(['{}-{}'.format(*t) for t in df.index], freq='M')
    return df


def m2m(df, start, n):
    """
    Return index for grouping starting at month 'start' and extending for 'n' months.
    The '0' group contains all the non-mathing months and needs to be dropped from groupby.
    """
    base = df.index.astype(int) - start + 1
    return (base % 12 < n).astype(int) * (base // 12 + 1970)

def year(df):
    c = df.copy()
    c.index = c.index.year
    return c

def model(dis, area):
    ago2mar = dis.groupby(m2m(dis,8,8)).mean().drop(0) # drop 0 group
    apr = year(dis[dis.index.month==4])
    y = ago2mar - 0.7 * apr

    apr2dec = area.groupby(m2m(area,4,9)).mean().drop(0)
    sep2nov = area.groupby(m2m(area,9,3)).mean().drop(0)
    x = (apr2dec + 0.3 * sep2nov) / np.max(area)
    return x,y

def plot(dis, area):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color cycle
    fig = plt.figure()
    area.plot(color='g')
    a = area.groupby(m2m(area,4,9)).mean().drop(0)
    b = area.groupby(m2m(area,9,3)).mean().drop(0)
    t = np.arange('2002','2017',dtype='datetime64[Y]') + np.timedelta64(3,'M')
    d = year(dis[dis.index.month==4])
    plt.bar(t.astype(datetime), a, 9, align='edge')
    plt.bar((t + np.timedelta64(5, 'M')).astype(datetime), b, 3, align='edge')
    area.plot(color='g')
    plt.gca().twinx()
    dis.plot(color='r')
    plt.bar(t[1:].astype(datetime), d, 1, align='edge', color='r')
    fig.show()

def model2(dis, area, total):
    fig = plt.figure()
    a = area.groupby(m2m(area,4,9)).mean().drop(0)
    d = dis.groupby(m2m(dis,8,8)).mean().drop(0)
    plt.scatter(np.log(1 - a[1:]/total), d[1:], c=d[:-1])
    fig.show()
