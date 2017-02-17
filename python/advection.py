#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import helpers as hh
import matplotlib.pyplot as plt
from mapping import basemap
import scipolate as sp


# nc = Dataset('wrfout_d02_2016-07-23_00:00:00')
# T = nc.variables['T'][:] + 300
# u = nc.variables['U'][:]
# v = nc.variables['V'][:]
# w = nc.variables['W'][:]
# gp = (nc.variables['PHB']+nc.variables['PH'])/9.81
# 
# uT = .5 * u[:,:,1:-1,1:-1]*(T[:,:,1:-1,:-1]+T[:,:,1:-1,1:])
# vT = .5 * v[:,:,1:-1,1:-1]*(T[:,:,:-1,1:-1]+T[:,:,1:,1:-1])
# wT = .5 * w[:,1:-1,1:-1,1:-1]*(T[:,:-1,1:-1,1:-1]+T[:,1:,1:-1,1:-1])
# 
# ad = np.diff(uT,1,3)/10000 + np.diff(vT,1,2)/10000
# dw = np.diff(wT,1,1)/np.diff(gp[:,1:-1,1:-1,1:-1],1,1)


D = pd.HDFStore('../data/station_data.h5')
S = pd.HDFStore('../data/LinearLinear.h5')

T = hh.extract(D['ta_c'],'prom',1)
Tm = S['T2']['d02']
sta = D['sta']
b = Tm-T

nc = Dataset('../data/wrf/d02_2014-09-10.nc')
t = pd.DatetimeIndex(hh.get_time(nc)) - np.timedelta64(4,'h')

ma = basemap(nc)
ij = ma(*hh.lonlat(sta))
lon,lat = ma.lonlat()
xy = (lon[1:-1,1:-1],lat[1:-1,1:-1])
ad = sp.grid_interp(xy, np.load('adv.npy'), ij, sta.index, t, method='linear')

d = pd.Panel({'adv':ad,'bias':b})
dm = d.groupby(d.major_axis.date,'major').mean().to_frame()