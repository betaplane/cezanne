#!/usr/bin/env python
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from geo import cells
import data

D = data.Data()
D.open('d','station_data.h5')
ra = D.d['ppa_mm'].xs('prom', 1, 'aggr')
r = D.d['pp_mm'].xs('prom', 1, 'aggr')
sta = D.sta.loc[r.columns.get_level_values('station')]

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


with xr.open_dataset('../../data/WRF/3d/geo_em.d03.nc') as d3:
    lm = d3['LANDMASK'].squeeze().load()
    lon, lat = d3['XLONG_C'].squeeze().load(), d3['XLAT_C'].squeeze().load()

i, j = cells(lon, lat, *sta[['lon', 'lat']].astype(float).values.T)

ds = xr.open_dataset('RAINNC_5days.nc')
x = ds['RAINNC'].isel(Time = np.arange(24, 121, 24))
xs = x.isel_points(south_north=i, west_east=j).mean('points')

xs = xr.concat((xs.isel(Time = 0), xs.diff('Time')), 'Time')


def offset_daily(x, delta='-8h'):
    """
    Resample hourly observations to daily, matching WRF simulation days (which start at 8:00h local time / previously 20:00h). The default -8h offset means the timestamp on the resampled series refers to the beginning of a 24h period starting at 8:00.
    """
    y = x.copy()
    y.index += pd.Timedelta(delta)
    return y.resample('D').mean() * 24

def loss(obs, mod, offset='MS'):
    o = obs.resample(offset).mean().mean(1)
    o.name = 'obs'
    def lead(n):
        m = mod.isel(Time=n)
        m['start'] = m.start + pd.Timedelta(n, 'd')
        return m.resample(offset, 'start', how='mean').to_series()
    r = np.arange(len(mod.Time))
    return pd.concat([lead(n) for n in r], 1, keys=r).dropna(0, 'all').join(o, how='inner')

lx = loss(offset_daily(r, '4h'), xs, 'D')


