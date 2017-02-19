#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipolate as sp
import mapping as mp
import helpers as hh

	

def interp_grib(grb, stations, method='linear', transf=False):
	"""
If transf=True, first project geographical coordinates to map, then use map coordinates
to interpolate using an unstructured array of points. Otherwise interpret lat/lon as regular
(enough) grid and use grid interpolation.
	"""
	x,t,lon,lat = hh.ungrib(grb)
	if transf:
		m = mp.basemap((lat,lon))
		return sp.irreg_interp(m(lon,lat), x, m(*hh.lonlat(stations)), stations.index, t, method=method)
	else: 
		lon -= 360
		return sp.grid_interp((lon,lat), x, hh.lonlat(stations), stations.index, t, method=method)


def interp_nc(nc, var, stations, time=True, tz=False, method='linear', map=None):
	m = mp.basemap(nc) if map is None else map
	xy = m.xy()
	ij = m(*hh.lonlat(stations))
	x = nc.variables[var][:].squeeze()
	if time:
		t = pd.DatetimeIndex(hh.get_time(nc))
		if tz:
			t = t.tz_localize('UTC').tz_convert(hh.CEAZAMetTZ())
		else:
			t -= np.timedelta64(4,'h')
		return sp.grid_interp(xy, x, ij, stations.index, t, method=method)
	else:
		return sp.grid_interp(xy, hh.g2d(x), ij, stations.index, method=method)



if __name__ == "__main__":
	from netCDF4 import Dataset
	
	d = Dataset('data/wrf/d02_2014-09-10.nc')
	D = pd.HDFStore('data/station_data.h5')
	sta = D['sta']
	x,z = interp_nc(d,'T2',sta)
	