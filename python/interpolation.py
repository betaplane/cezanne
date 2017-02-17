#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate as ip
from mapping import basemap, affine
import helpers as hh


def grid_interp(xy, data, ij, index, time=None, method='nearest'):
    x, y = xy
    i, j = ij
    m, n = data.shape[-2:]
    tg = affine(x, y)
    coords = np.roll(tg(np.r_['0,2', [i, j]]).T, 1, 1)
    if time is None:
        return pd.Series(
            ip.interpn(
                (range(m), range(n)), data, coords, method,
                bounds_error=False),
            index=index)
    else:
        return pd.DataFrame(
         [ip.interpn((range(m),range(n)),data[k,:,:],coords,method,bounds_error=False) \
          for k in range(len(time))],
         index=time, columns=index
        )


def interp4D(xy, data, ij, index, t, method='nearest'):
    x, y = xy
    i, j = ij
    mn = (tuple(range(data.shape[-2])), tuple(range(data.shape[-1])))
    tg = affine(x, y)
    coords = np.roll(tg(np.r_['0,2', [i, j]]).T, 1, 1)
    a = dict([(l,dict([(t[k], ip.interpn(mn, data[k,l,:,:], coords, method, bounds_error=False)) \
     for k in range(data.shape[0])])) for l in range(data.shape[1])])
    p = pd.Panel(a).transpose(1, 2, 0)
    p.items = index
    return p


def irreg_interp(xy, data, ij, index, time=None, method='nearest'):
    x, y = xy
    i, j = ij
    interpolator = {
        'nearest': ip.NearestNDInterpolator,
        'linear': ip.LinearNDInterpolator
    }
    if time is None:
        itrp = interpolator[method]((x.flatten(), y.flatten()), data.flatten())
        return pd.Series(itrp(i, j), index=index)
    else:
        df = []
        for k in range(len(time)):
            print(k)
            itrp = interpolator[method]((x.flatten(), y.flatten()),
                                        data[k, :, :].flatten())
            df.append(itrp(i, j))
        return pd.DataFrame(df, index=time, columns=index)



def interp_grib(grb, stations, method='linear', transf=False):
    """
If transf=True, first project geographical coordinates to map, then use map coordinates
to interpolate using an unstructured array of points. Otherwise interpret lat/lon as regular
(enough) grid and use grid interpolation.
	"""
    x, t, lon, lat = hh.ungrib(grb)
    if transf:
        m = basemap((lat, lon))
        return irreg_interp(
            m(lon, lat),
            x,
            m(*hh.lonlat(stations)),
            stations.index,
            t,
            method=method)
    else:
        lon -= 360
        return grid_interp(
            (lon, lat),
            x,
            hh.lonlat(stations),
            stations.index,
            t,
            method=method)


def interp_nc(nc,
              var,
              stations,
              time=True,
              tz=False,
              method='linear',
              bmap=None):
    m = basemap(nc) if bmap is None else bmap
    xy = m.xy()
    ij = m(*hh.lonlat(stations))
    x = nc.variables[var][:].squeeze()
    if time:
        t = pd.DatetimeIndex(hh.get_time(nc))
        if tz:
            t = t.tz_localize('UTC').tz_convert(hh.CEAZAMetTZ())
        else:
            t -= np.timedelta64(4, 'h')
        return grid_interp(xy, x, ij, stations.index, t, method=method)
    else:
        return grid_interp(xy, hh.g2d(x), ij, stations.index, method=method)


if __name__ == "__main__":
    from netCDF4 import Dataset

    d = Dataset('data/wrf/d02_2014-09-10.nc')
    D = pd.HDFStore('data/station_data.h5')
    sta = D['sta']
    x, z = interp_nc(d, 'T2', sta)
