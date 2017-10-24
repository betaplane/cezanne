#!/usr/bin/env python
import xarray as xr
import numpy as np
from pyproj import Proj, Geod
from geo import proj_params
from interpolation import grid_interp
from subprocess import Popen, PIPE
import os
from datetime import datetime
from functools import partial

def time(d):
    pr = Popen(['../../data/WRF/op.sh', str(d)], stdout=PIPE)
    f1, err = pr.communicate()
    t1 = np.array([datetime.strptime(os.path.split(f)[1], 'wrfout_d03_%Y-%m-%d_%H:%M:%S')
                   for f in f1.decode().splitlines()], dtype='datetime64[h]')
    return (np.arange(0, 24, dtype='timedelta64[h]').reshape((-1, 1)) + t1).flatten('F')


with xr.open_dataset('../../data/WRF/3d/geo_em.d03.nc') as g:
    p = Proj(**proj_params(g))
    lon, lat = g['XLONG_M'].squeeze().values, g['XLAT_M'].squeeze().values
    ll = p(lon, lat)

ij = np.array([[-70.33290088, -30.543625], [-70.539993, -31.262951]])

# with xr.open_dataset('../../data/WRF/sw0.nc') as ds:
#     t = time(0)
#     x = ds['SWDOWN'].squeeze().values
#     y0 = grid_interp(ll, x, ij.T, ['Guandacol', 'Tascadero'], t, method='linear')

# with xr.open_dataset('../../data/WRF/sw1.nc') as ds:
#     t = time(1)
#     x = ds['SWDOWN'].squeeze().values
#     y1 = grid_interp(ll, x, ij.T, ['Guandacol', 'Tascadero'], t, method='linear')

# y = pd.concat((y0[y0.index.hour < 12], y1[y1.index.hour > 11]), 0).sort_index()

def dist(lon, lat):
    inv = partial(Geod(ellps='WGS84').inv, lon, lat)

    def dist(x, y):
        return inv(x, y)[2]

    return np.vectorize(dist)


gmask = np.exp(-(dist(*ij[1])(lon, lat) / 20000) ** 2)

with xr.open_dataset('../../data/WRF/ppt0.nc') as ds:
    t0 = time(0)
    x0 = (ds['RAINNC'].squeeze().values * gmask).sum(1).sum(1) / gmask.sum()
    y0 = pd.DataFrame(x0, index=t0)

with xr.open_dataset('../../data/WRF/ppt1.nc') as ds:
    t1 = time(1)
    x1 = (ds['RAINNC'].squeeze().values * gmask).sum(1).sum(1) / gmask.sum()
    y1 = pd.DataFrame(x1, index=t1)

y = pd.concat((y0[y0.index.hour < 12], y1[y1.index.hour > 11]), 0).sort_index().diff(1, 0)
y0 = y0.diff(1, 0)
y.loc[y.index.hour==0, :] = y0[y0.index.hour==0]
