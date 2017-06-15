#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
from collections import Counter

# Shafer, Mark A., Christopher A. Fiebrich, Derek S. Arndt, Sherman E. Fredrickson, and Timothy W. Hughes. “Quality Assurance Procedures in the Oklahoma Mesonetwork.” Journal of Atmospheric and Oceanic Technology 17, no. 4 (2000): 474–494.

qp = pd.DataFrame(index = D.sta.index)
qp[('ta_c', 'range', 'min')] = -25
qp[('ta_c', 'range', 'max')] = 40
qp.columns = pd.MultiIndex.from_tuples(qp.columns)

class QC(object):
    def __init__(self, data, var):
        self.data = data

    @staticmethod
    def _xr(x, name):
        y = xr.DataArray(x).expand_dims('check')
        y['check'] = [name]
        return y.astype(float).fillna(0)

    def range(self, var):
        "values that fall outside climatological range"
        MI, mi = data.xs('min', 1, 'aggr').align(qp[(var, 'range', 'min')], axis=1, level='station', broadcast_axis=0)
        MA, ma = data.xs('max', 1, 'aggr').align(qp[(var, 'range', 'max')], axis=1, level='station', broadcast_axis=0)
        qf = _xr(MI < mi, 'min')
        qf = xr.concat((qf, _xr(MA > ma, 'max')), 'check')
        qf = xr.concat((qf, _xr(abs(data.xs('avg', 1, 'aggr') - MA + MI), 'avg')), 'check')

    def _step(y, max_ts):
        x = y.dropna()
        dt = np.diff(np.array(x.index, dtype='datetime64[m]').astype(int))
        i = (dt <= max_ts)
        delta = x.iloc[1:][i].as_matrix() / dt[i].reshape((-1, 1))
        return pd.DataFrame(delta, index=y.index[1:][i], columns=y.columns)

    def step(self):
        "change between successive observations"
        data.groupby(axis=1, level='station')

    def persistence(self):
        pass

    def spatial(self):
        pass
