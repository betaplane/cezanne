#!/usr/bin/env python
import numpy as np
import pandas as pd

D = pd.HDFStore('../../data/tables/station_data_new.h5')
rs = D['rs_w']
Ra = D['Ra_w'][rs.columns]
drop_codes = ['28', 'ANDACMP10', 'CACHRSTM', 'COMBRS', 'INILLARS', 'MINRS', 'PCRS2', 'TLHRSTM']

dist = D['dist'].loc[rs.columns, rs.columns]

def diff(x):
    d = rs[dist[x.name].sort_values().index[1:5]].apply(lambda y:abs(y-x))
    ds = (d/d.std()).sum(1)


def detect_breaks(df):
    i = df.dropna().index
    d = pd.Series(np.diff(i), index=i[1:])
    dd = d[d>np.timedelta64(1,'D')].sort_index()
    return dd.index[-1] if len(dd) else np.nan

rs.apply(lambda c: np.sort(np.diff(c.dropna().index))[-1],0)

rs[Ra>0]>Ra
