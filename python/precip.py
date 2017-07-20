#!/usr/bin/env python
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from datetime import datetime
import helpers as hh
import data

D = data.Data()
D.open('d','station_data.h5')
ra = D.d['ppa_mm'].xs('prom', 1, 'aggr')
r = D.d['pp_mm'].xs('prom', 1, 'aggr')
# a = pd.read_csv('../../data/CEAZAMet/Marion/LaLaguna_hourly_precipitations.csv', sep=';')
# b = pd.read_csv('../../data/CEAZAMet/Marion/Tapado_hourly_precipitations.csv', sep=';')
# c = pd.read_csv('../../data/CEAZAMet/Marion/LlanoHuanta_hourly_precipitations.csv', sep=';')
# m = ['year', 'month', 'day', 'hour']
# for d in [a, b, c]:
#     d.index = [datetime(*i.loc[m].astype(int)) for j, i in d.iterrows()]
#     d.drop(m, 1, inplace=True).replace(-9999, np.nan)

# def clean(x):
#     d = np.diff(ra.values, 1, 0)
#     i, j = np.where((d[:-1, :] != 0) & (d[:-1, :] == -d[1:, :]))
#     return [i[np.where(j == k)[0]] + 1 for k in sorted(set(j))]

# def diff(x):
#     d = x.dropna().diff(1)
#     d[d < 0] = np.nan
#     return d

ds = xr.open_dataset('../../data/WRF/3d/RAINNC_6days.nc')
x = ds['RAINNC'].diff('Time')

Map = hh.basemap()
i, j = Map(ds.XLONG[0,:,:].values, ds.XLAT[0,:,:].values)
sta = D.sta.loc[r.columns.get_level_values('station')]
lon, lat = sta[['lon','lat']].astype(float).values.T

cm = plt.get_cmap('Set2')

def run_base(x_mod, x_obs, ndays, levels, k):
    plt.gca().collections = []
    plt.gca().artists = []
    lgnd = []
    for t, s in zip(*np.where(x_mod.XTIME == k)):
        r = x_mod.isel(start = s, Time = t)
        d = r.start.values.astype('datetime64[D]').astype(int) % ndays
        m = Map.contour(i, j, r, colors = [cm(d)], levels = levels)
        c = m.collections[0].get_edgecolor()[0]
        lgnd.append((plt.Rectangle((0, 0), 1, 1, fc = c), r.start.values.astype('datetime64[h]')))
    plt.title(k)
    plt.gca().add_artist(
        plt.legend(*zip(*sorted(lgnd, key=lambda s:s[1])), loc='center left', bbox_to_anchor=(1, .5))
    )
    Map.drawcoastlines()
    Map.drawcountries()
    o = x_obs.loc[k].values
    l = np.where(o == 0)[0]
    m = np.where(o > 0)[0]
    Map.scatter(lon[l], lat[l], c='gray', marker= 'x', latlon=True)
    Map.scatter(lon[m], lat[m], o[m]*10, c='r', marker= 'x', latlon=True)
    plt.legend([plt.scatter([], [], s, c='r', marker='x') for s in [10, 20, 50]],
               ['1 mm', '2 mm', '5 mm'], loc='upper left', bbox_to_anchor=(1, 1))
    return lgnd

run = partial(run_base, x, r, 6, [.1])

frames = np.arange('2016-10-11T09','2017-07-16T09',dtype='datetime64[h]')
# Frames = np.arange('2017-07-01T13','2017-07-02T13',dtype='datetime64[h]')

an = FuncAnimation(plt.gcf(), run, frames, interval=200, repeat=False)
an.save('test.mp4',writer='avconv', codec='libx264')
