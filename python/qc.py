#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr

# Shafer, Mark A., Christopher A. Fiebrich, Derek S. Arndt, Sherman E. Fredrickson, and Timothy W. Hughes. “Quality Assurance Procedures in the Oklahoma Mesonetwork.” Journal of Atmospheric and Oceanic Technology 17, no. 4 (2000): 474–494.

qp = pd.DataFrame(index = D.sta.index)
qp[('ta_c', 'min')] = -25
qp[('ta_c', 'max')] = 40
qp.columns = pd.MultiIndex.from_tuples(qp.columns)

class QC(object):
    def __init__(self, data, var):
        self.data = data

    def range(self, var):
        "Values that fall outside climatological range."
        MI, mi = data.xs('min', 1, 'aggr').align(qp[var]['min'], axis=1, level='station', broadcast_axis=0)
        MA, ma = data.xs('max', 1, 'aggr').align(qp[var]['max'], axis=1, level='station', broadcast_axis=0)
        qf = xr.DataArray(MI < mi).astype(float).fillna(0)
        qf = xr.concat((qf, xr.DataArray(MA > ma).astype(float).fillna(0)), pd.Index(['min', 'max'], name='check'))


    def step(self):
        pass

    def persistence(self):
        pass

    def spatial(self):
        pass
