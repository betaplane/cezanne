#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import helpers as hh


class basemap(Basemap):
    def __init__(self, obj):
        def lonlat(lon, lat):
            from pyproj import Geod
            geo = Geod(ellps='WGS84')
            x = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]
            return {
                'lon_0': .5 * np.sum(x[:2]),
                'lat_0': .5 * np.sum(x[2:]),
                'width': geo.inv(x[0], x[3], x[1], x[3])[2],
                'height': geo.inv(x[0], x[2], x[0], x[3])[2],
                'lat_1': -10,
                'lat_2': -40
            }

        def atts(file, width=None, height=None):
            return {
                'lat_0':
                file.CEN_LAT,
                'lat_1':
                file.TRUELAT1,
                'lat_2':
                file.TRUELAT2,
                'lon_0':
                file.CEN_LON,
                'width':
                file.DX * (getattr(file, 'WEST-EAST_GRID_DIMENSION')
                           if width is None else width),
                'height':
                file.DY * (getattr(file, 'SOUTH-NORTH_GRID_DIMENSION')
                           if height is None else height)
            }

        def attshw(file):
            h, w = file.variables['XLONG_M'].shape[-2:]
            return atts(file, w, h)

        super().__init__(projection='lcc',
         **hh.try_list(obj,
          lambda x: lonlat(x['lon'],x['lat']),
          lambda x: x.to_dict(),
          lambda x: hh.try_list(x, atts, attshw,
           lambda x: lonlat(hh.g2d(x.variables['XLONG']),hh.g2d(x.variables['XLAT'])),
          ),
          lambda x: lonlat(*x),
          lambda x: lonlat((lambda a,b:b,a)(*x[1].latlons()))
         )
        )
        try:
            self._lonlat = hh.lonlat(obj)
        except:
            pass

    def xy(self):
        return self(*self._lonlat)

    def affine(self):
        return affine(*self(*self._lonlat))


def map_plot(df, sta, map=None):
    if map is None:
        map = basemap(sta.loc[df.dropna().index])
    elif not isinstance(map, Basemap):
        map = basemap(map)
    fig = plt.figure()
    try:
        B = pd.concat(
            sta.loc[:, ('lon', 'lat')].align(df, level='station', axis=0),
            axis=1).dropna()
    except:
        B = pd.concat(
            sta.loc[:, ('lon', 'lat')].align(df, axis=0), axis=1).dropna()
    norm = colors.Normalize(vmin=min(B.iloc[:, -1]), vmax=max(B.iloc[:, -1]))
    sm = cm.ScalarMappable(norm=norm)
    sm.set_array(B.iloc[:, -1])
    for i, c in B.iterrows():
        map.plot(
            c['lon'], c['lat'], 'o', color=sm.to_rgba(c.iloc[-1]), latlon=True)
    plt.colorbar(sm)
    map.drawcoastlines()
    map.drawparallels(range(-40, -15, 5), labels=[1, 1, 1, 1])
    map.drawmeridians(range(-90, -60, 5), labels=[1, 1, 1, 1])
    fig.show()
    return map


def affine(x, y):
    m, n = x.shape
    j, i = np.mgrid[:m, :n]

    C = np.linalg.lstsq(
        np.r_[[x.flatten()], [y.flatten()], np.ones((1, m * n))].T,
        np.r_[[i.flatten()], [j.flatten()], np.ones((1, m * n))].T)[0].T
    b = C[:2, 2]
    A = C[:2, :2]

    def to_grid(coords):
        return A.dot(coords) + np.r_[[b]].T

    return to_grid


if __name__ == "__main__":
    from netCDF4 import Dataset
    import pandas as pd
    import matplotlib.pyplot as plt

    store = pd.HDFStore('data/data.h5')
    sta = store['sta']

    g = Dataset('data/wrf/geo_em.d02.nc')
    lon = g.variables['XLONG_M'][0, :, :]
    lat = g.variables['XLAT_M'][0, :, :]
    z = g.variables['HGT_M'][0, :, :]

    map = basemap(g)
    to_grid = affine(*map(lon, lat))

    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.pcolormesh(z)
    for i, s in sta.iterrows():
        i, j = to_grid(*map(s['lon'], s['lat']))
        plt.plot(i, j, 'xw')
    ax = plt.subplot(1, 2, 2)
    map.pcolormesh(lon, lat, z, latlon=True)
    map.plot(sta['lon'].tolist(), sta['lat'].tolist(), 'xw', latlon=True)
    fig.show()
