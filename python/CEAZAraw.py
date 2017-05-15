#!/usr/bin/env python
import numpy as np
import pandas as pd
import helpers as hh
from collections import Counter


def group2idx(df):
    "Return a DataFrame with proper index from one with groupby((index.date, index.hour))."
    df.index= pd.DatetimeIndex([np.datetime64(i[0]) + np.timedelta64(i[1], 'h') for i in df.index])
    return df

def tshift(df, h=-1):
    d = df.copy()
    d.index += pd.Timedelta(h, 'h')
    return d

def decol(df):
    d = df.copy()
    d.columns = [0]
    return d

def hour_ave(df):
    """
    Average raw data by true hour. Takes a one-column DataFrame or Series.
    Record period (i.e. the time interval over which data has been accumulated)
    is deduced from Timestamp differences - hence concatenated DataFrames are
    no use. Returns (df, idx), where df is the averaged DataFrame and idx is an index
    containing all times where missing data intervals where backfilled.
    """
    dt = np.diff(df.index).astype(float) * 1e-9
    c = Counter(dt) # counts distinct elements in dt
    # heuristic for cleaning:
    # eliminate elements whose occurrence is less than half of the time over which they do occur
    f = lambda i: not (1 <= (max(i[0]) - min(i[0])) / i[1] < 2)
    g = list(filter(f, [(np.where(dt==k)[0], n, k) for k,n in c.items()]))
    print('all counts:')
    [print(i) for i in c]
    print('')
    print('filtered:')
    [print(i) for i in g]

    x = pd.DataFrame({'idx': df.index, 'dt': np.r_[np.nan, dt]}, index=df.index).join(decol(df))
    idx = x.index[x['dt'].isin([i[-1] for i in g])]
    # eleminate all the 'dt' instances arising from data gaps
    x.loc[idx, 'dt'] = np.nan # equivalent to sql 'in' or general 'contains'
    # backfill the eliminated 'dt'
    x['dt'] = x['dt'].fillna(method='bfill')

    # extract first record of the hour
    i1 = group2idx(x.groupby((x.index.date, x.index.hour)).apply(lambda h: h.iloc[0]))
    # get indexes of rest
    ir = pd.Index(set(x.index) - set(i1['idx'])).sort_values()

    # minutes of record which belong to 'current' hour
    curr = i1.apply(lambda h: min(h['idx'].minute, h['dt']), 1)
    # minutes belonging to previous hour
    prev = i1['dt'] - curr

    # group remaining records by the hour and form numerator and denominator for average
    gr = x.loc[ir].groupby((ir.date, ir.hour))
    num = group2idx(gr.apply(lambda h: np.sum(h.iloc[:,-1] * h['dt'])))
    den = group2idx(gr['dt'].sum())

    # add values from first hourly record to numerator and denominator
    num = num.add(i1.iloc[:,-1] * curr, fill_value=0)
    num = num.add(tshift(i1.iloc[:,-1] * prev), fill_value=0)
    den = den.add(curr, fill_value=0)
    den = den.add(tshift(prev), fill_value=0)

    return pd.DataFrame(num / den, columns = df.columns), idx
