#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os, pygrib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from interpolation import interp_nc, interpolate, interp_grib, get_nearest
from itertools import combinations

store = pd.HDFStore('data/data.h5')

sta = store['sta']
dist = store['dist']

# weight matrix by using sum of squared distances to all other stations
# heuristic density ~ area ~ dist**2
d = dist.apply(lambda x: x**2).sum()

# remove far away stations
D = d.loc[d < d.mean() + d.std()]
D = D / D.sum()

T = store['ta_c'].xs('prom', level='aggr', axis=1)
T = T.drop(T.xs(10, level='elev', axis=1), axis=1)
T.columns = T.columns.get_level_values('station')

T2 = store['T02']

T.drop('CGR', axis=1, inplace=True)  # no overlap, makes it easier below

bias = T2 - T
bias_anom = bias - bias.mean()
B = bias * D
cov = B.cov().dropna(how='all').dropna(axis=1, how='all')
s = cov.index


def eof(cov, n=5):
    w, v = np.linalg.eig(cov)
    i = np.argsort(w)[::-1]
    print(100 * w[i[:n]] / np.sum(w))
    return pd.DataFrame(v[:, i[:n]], index=cov.index)


eofs = eof(cov)
pcs = B[s].dot(eofs)

rcst = pcs.dot(eofs.transpose()) / D[s]

def comb(m):
    success = False
    i = []
    for l in range(len(m))[::-1]:
        for j, c in enumerate(combinations(m.index, l)):
            print('{} {}'.format(l, j))
            i = list(c)
            if m[i].loc[i].isnull().sum().sum() == 0:
                success = True
                break
        if success:
            break
    return i

# fig = plt.figure()
# plt.plot_date(B.index,pc,'-')
# fig.show()
