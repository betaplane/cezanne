#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate as ip
from mapping import basemap, affine, matts
from mpl_toolkits.basemap import Basemap as bmap
import helpers as hh
import xarray as xr
import re


def grid_interp(xy, data, ij, index, time=None, method='linear'):
    "Interpolation for data on a sufficiently regular mesh."
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


def irreg_interp(xy, data, ij, index, time=None, method='linear'):
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


def interp4D(xy, data, ij, index, t, method='linear'):
    "Interpolates 4D data on 2D grids (i.e. no interpolation in vertical)."
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


def grib_interp(grb, stations, method='linear', transf=False):
    """
If transf=True, first project geographical coordinates to map, then use map coordinates
to interpolate using an unstructured array of points. Otherwise interpret lat/lon as regular
(enough) grid and use grid interpolation.
	"""
    x, t, lon, lat = hh.ungrib(grb)
    if transf:
        m = basemap((lon, lat))
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


def nc_interp(nc,
              var,
              stations,
              time=True,
              tz=False,
              method='linear',
              map=None):
    m = basemap(nc) if map is None else map
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


def xr_interp(nc,
              var,
              stations,
              time=True,
              method='linear',
              map=None,
              dt=-4):
    with xr.open_dataset(nc) as ds:
        m = bmap(projection='lcc', **matts(ds)) if map is None else map
        ij = m(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
        v = ds[var]
        lon, lat, time = hh.coord_names(v, 'lon', 'lat', 'time')
        xy = m(hh.g2d(v.coords[lon]), hh.g2d(v.coords[lat].data))
        x = np.array(ds[var]).squeeze()
        if time:
            t = ds['XTIME'] + np.timedelta64(dt, 'h')
            df = grid_interp(xy, x, ij, stations.index, t, method=method)
        else:
            df = grid_interp(xy, hh.g2d(x), ij, stations.index, method=method)
    return (df, m) if map is None else df

# Below are some implementations of interpolations of 3D fields
# to 1D vertical profiles. I have compared the results to simplified
# procedures (interpolation in horizontal first, e.g. with interp4D)
# and the differences don't seem to warrant the more expensive approach.

# 3D interpolation using only 4 surrounding points
def int4(map, k, sta):
    "t4 = int4(map, time_index, sta['CIM00085586'])"
    x, y = map.xy()

    def xy(i, j):
        return (x[i, j], y[i, j])

    def contains(p):
        for i in range(lon.shape[0] - 1):
            for j in range(lon.shape[1] - 1):
                if Polygon((xy(i, j), xy(i + 1, j), xy(i + 1, j + 1), xy(
                        i, j + 1), xy(i, j))).contains(p):
                    return i, j

    lon, lat = map(*sta[['lon', 'lat']])
    i, j = contains(Point(lon, lat))
    d = np.array(
        (xy(i, j), xy(i + 1, j), xy(i + 1, j + 1), xy(i, j + 1))).repeat(29, 0)
    Tx = np.array((T[k, :, i, j], T[k, :, i + 1, j], T[k, :, i + 1, j + 1],
                   T[k, :, i, j + 1]))
    Px = np.array((P[k, :, i, j], P[k, :, i + 1, j], P[k, :, i + 1, j + 1],
                   P[k, :, i, j + 1]))
    d = np.r_['0,2', d.T, np.log([Px.flatten()])]
    l = np.r_['0,2', np.repeat([(lon, lat)], len(pl), 0).T, [pl]]
    return LinearNDInterpolator(d.T, Tx.flatten(), rescale=True)(l.T)


# 3D interpolation on complete unstructured mesh
def inta(j, k):
    "ta = inta(0,t[0])"
    d = np.r_[np.r_[[x.flatten()], [y.flatten()]].repeat(29, 1),
              [np.log(P[j, :, :, :]).transpose((1, 2, 0)).flatten()]]
    l = np.r_['0,2', np.r_[[ij[0]], [ij[1]]].repeat(len(pl), 1),
              [np.repeat([pl], 2, 0).flatten()]]
    return LinearNDInterpolator(
        d.T, T[j, :, :, :].transpose(
            (1, 2, 0)).flatten(), rescale=True)(l.T).reshape((2, -1))

if __name__ == "__main__":
    from netCDF4 import Dataset

    d = Dataset('data/wrf/d02_2014-09-10.nc')
    D = pd.HDFStore('data/station_data.h5')
    sta = D['sta']
    x, z = interp_nc(d, 'T2', sta)
