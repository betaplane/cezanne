#!/usr/bin/env python
import pandas as pd
import numpy as np

D = pd.HDFStore('../../data/tables/station_data_new.h5')
vv = D['vv_ms'].xs('prom',1,'aggr')
vv.columns = vv.columns.get_level_values('code')

m = vv['2010':]
m = m.loc[:, m.count() > .7 * len(m)]
n = m.copy()
n[:] = 0
n[m.isnull()] = 1
y = m.fillna(0) + n * m.mean()

w, v = np.linalg.eig(y.cov())
u = v[:,0]
t = y.dot(v)
r = t.dot(v)
r.columns = y.columns
r += y.mean()
# r = pd.DataFrame(np.r_['1,2',t].T*u, index=t.index, columns=m.columns)
y = m.fillna(0) + n * r
