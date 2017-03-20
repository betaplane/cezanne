#!/usr/bin/env python
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import CubicSpline
import helpers as hh


nc = Dataset('../../data/NCEP/X164.77.119.34.78.12.40.44.nc')
t = hh.get_time(nc)
lon,lat = hh.lonlat(nc)
slp = nc.variables['slp'][:]

D = pd.HDFStore('../../data/tables/station_data_new.h5')
v = D['vv_ms']

x = v['4']
xd = x.groupby(x.index.date).mean()
xd.columns = x.columns.get_level_values('elev')
xd.index = pd.DatetimeIndex(xd.index
x5 = xd['5']['2012-07-23']
