#!/usr/bin/env python
import pandas as pd
import numpy as np
from ncep_corr import dti, tfloat, CubicSpline


D = pd.HDFStore('../../data/tables/station_data_new.h5')
vv = D['vv_ms'].xs('prom',1,'aggr')

# remove multiple sensors at same station
vv.drop(['PAZVV5', '117733106', 'RMRVV5', 'VCNVV5', 'PEVV5', 'RMPVV5', 'CGRVV10',
         'CHPLVV10', 'LCARVV5', 'LLHVV10M', 'QSVV1', 'VLLVV30'], 1, 'code', inplace=True)
vv.columns = vv.columns.get_level_values('station')

# remove suspicious data
vv.drop('9', 1, inplace=True)
vv['INIA66'][:'2012-03-20'] = np.nan

d = vv['2013':]
d = d.loc[:, d.count() > .7 * len(d)]
dm = dti(d.groupby(d.index.date).mean())

dist = D['dist'][d.columns].loc[d.columns]
sta = D['sta'].loc[d.columns]

from helpers import basemap
m = basemap()
m.plot(*sta.loc[d.columns][['lon','lat']].as_matrix().T, 'o', latlon=True)
m.drawcoastlines()

def spline(x):
    m = dti(x.groupby((x.index.year, x.index.month)).mean()).dropna()
    i = tfloat(m.index[1:-1] + np.diff(m.index[1:]) / 2)
    cs = CubicSpline(i, m[1:-1], bc_type='natural')
    y = x[m.index[1]:m.index[-1]]
    return pd.DataFrame(y.as_matrix() - cs(tfloat(y.index, '12H')), index=y.index)

s = pd.concat([spline(x) for i,x in dm.iteritems()], 1)
s.columns = dm.columns
s -= s.mean()

def mask(d):
    m = d.copy()
    m[:] = 0
    m[d.isnull()] = 1
    return m


def pca(d, r=None, n=d.shape[1], scale=False):
    m = mask(d)
    if r is None:
        y = d.fillna(0)
    else:
        y = d.fillna(0) + m * r
    c = d.cov()
    if scale:
        c *= dist
    w, v = np.linalg.eig(c)
    i = np.argsort(w) # eigenvalues not guaranteed to be ordered
    t = y.dot(v[:,i[:n]])
    r = t.dot(v[:,i[:n]].T)
    r.columns = d.columns
    mse = ((r-d)**2).sum().sum()
    print(mse)
    # r = pd.DataFrame(np.r_['1,2',t].T*u, index=t.index, columns=m.columns)
    return t.iloc[:,i[-1]], r

