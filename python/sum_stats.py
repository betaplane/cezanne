#!/usr/bin/env python
import pandas as pd
import data
import helpers as hh
from CEAZAMet import binning

D = data.Data()
D.open('d', '_data.h5')
D.open('r', '_raw.h5')
D.open('s', 'Linear')

def xtr(x, aggr):
    return hh.stationize(x.xs(aggr, 1, 'aggr').drop('10', 1, 'elev')) + 273.15

def bin(d, var, *args, **kwargs):
    v = lambda x: x.xs(var, 1, 'field', False)
    return pd.concat([binning(v(x), *args, **kwargs) for k, x in dict(d).items()], 1)

Tm = D.s['T2']
T = xtr(D.d['ta_c'], 'prom')

b = xtr(bin(D.r, 'ta_c'), 'avg')
