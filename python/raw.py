#!/usr/bin/env python
import numpy as np
import pandas as pd
import helpers as hh


def gr2idx(df):
    df.index= pd.DatetimeIndex([np.datetime64(i[0]) + np.timedelta64(i[1], 'h') for i in df.index])
    return df

def tshift(df, h=-1):
    df.index += pd.Timedelta(h, 'h')
    return df

def hour_ave(df):
    """
    Average raw data by true hour. Takes a one-column DataFrame or Series.
    Record period (i.e. the time interval over which data has been accumulated)
    is deduced from Timestamp differences - hence concatenated DataFrames are
    no use. 
    """
    x = pd.DataFrame({'idx': df.index[1:],
                      'dt': np.diff(df.index).astype('timedelta64[m]').astype(int)
                  }, index=df.index[1:]).join(df)

    # extract first record of the hour
    i1 = gr2idx(x.groupby((x.index.date, x.index.hour)).apply(lambda h: h.iloc[0]))
    # get indexes of rest
    ir = pd.Index(set(x.index) - set(i1['idx'])).sort_values()

    # minutes of record which belong to 'current' hour
    curr = i1.apply(lambda h: min(h['idx'].minute, h['dt']), 1)
    # minutes belonging to previous hour
    prev = i1['dt'] - curr

    # group remaining records by the hour and form numerator and denominator for average
    gr = x.loc[ir].groupby((ir.date, ir.hour))
    num = gr2idx(gr.apply(lambda h: np.sum(h.iloc[:,-1] * h['dt'])))
    den = gr2idx(gr['dt'].sum())

    # add values from first hourly record to numerator and denominator
    num = num.add(i1.iloc[:,-1] * curr, fill_value=0)
    num = num.add(tshift(i1.iloc[:,-1] * prev), fill_value=0)
    den = den.add(curr, fill_value=0)
    den = den.add(tshift(prev), fill_value=0)

    return num / den
