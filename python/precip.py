#!/usr/bin/env python
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import helpers as hh
import data

D = data.Data()
D.open('d','station_data.h5')
ra = D.d['ppa_mm'].xs('prom', 1, 'aggr')

def clean(x):
    d = np.diff(ra.values, 1, 0)
    i, j = np.where((d[:-1, :] != 0) & (d[:-1, :] == -d[1:, :]))
    return [i[np.where(j == k)[0]] + 1 for k in sorted(set(j))]

def diff(x):
    d = x.dropna().diff(1)
    d[d < 0] = np.nan
    return d

ds = xr.open_dataset('../../data/WRF/3d/RAINNC_6days.nc')
x = ds['RAINNC'].diff('Time')

Map = hh.basemap()
i, j = Map(ds.XLONG[0,:,:].values, ds.XLAT[0,:,:].values)

cm = plt.get_cmap('Set1')

def run_base(x, Map, ndays, levels, k):
    plt.gca().collections = []
    lgnd = []
    for t, s in zip(*np.where(x.XTIME == k)):
        r = x.isel(start = s, Time = t)
        d = r.start.values.astype('datetime64[D]').astype(int) % ndays
        m = Map.contour(i, j, r, colors = [cm(d)], levels = levels)
        c = m.collections[0].get_edgecolor()[0]
        lgnd.append((plt.Rectangle((0, 0), 1, 1, fc = c), r.start.values.astype('datetime64[h]')))
    plt.title(k)
    plt.legend(*zip(*sorted(lgnd, key=lambda s:s[1])), loc='center left', bbox_to_anchor=(1, .5))
    Map.drawcoastlines()
    return lgnd

run = partial(run_base, x, Map, 6, [.1])

frames = np.arange('2016-10-11T13','2017-07-16T13',dtype='datetime64[h]')
# frames = np.arange('2017-07-01T13','2017-07-02T13',dtype='datetime64[h]')

an = FuncAnimation(plt.gcf(), run, frames, interval=500, repeat=False)
an.save('test.mp4',writer='avconv', codec='libx264')
