#!/usr/bin/env python
from simplekml import Kml, Style
from helpers import nearest
from shapely.geometry import Polygon, Point, MultiPoint, LinearRing
import numpy as np


def kml(name, lon, lat, code=None, nc=None):
    if nc is not None:
        x = nc.variables['XLONG_M'][0,:,:]
        y = nc.variables['XLAT_M'][0,:,:]
        xc = nc.variables['XLONG_C'][0,:,:]
        yc = nc.variables['XLAT_C'][0,:,:]

    k = Kml()
    z = zip(name, lon, lat) if code is None else zip(name, lon, lat, code)
    for s in z:
        p = k.newpoint(name = s[3] if len(s)==4 else s[0], coords = [s[1:3]])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        p.style.balloonstyle.text = s[0]
        if nc is not None:
            i,j,d = nearest(x, y, s[1], s[2])
            coords = [
                (xc[i,j],yc[i,j]),
                (xc[i,j+1],yc[i,j]),
                (xc[i,j+1],yc[i+1,j]),
                (xc[i,j],yc[i+1,j]),
                (xc[i,j],yc[i,j])
            ]
            if Polygon(coords).contains(Point(*s[1:3])):
                l = k.newlinestring(coords = [s[1:3], (x[i, j], y[i, j])])
                r = k.newlinestring(coords=coords)
    return k


def cells(grid_lon, grid_lat, lon, lat, mask=None):
    """Get grid indexes corresponding to lat/lon points, using shapely polygons.

    :param grid_lon: grid of corner longitudes ('LONG_C' in geo_em... file)
    :param grid_lat: grid of corner latitudes ('LAT_C' in geo_em... file)
    :param lon: array of point longitudes
    :param lat: array of point latitudes
    :param mask: 1-0 mask of grid points to be taken into account (e.g. land mask)
    :returns: i, j arrays of grid cell indexes

    """
    s = np.array(grid_lon.shape) - 1
    if mask is None:
        def ll(i, j):
            return np.r_[grid_lon[i:i+s[0], j:j+s[1]], grid_lat[i:i+s[0], j:j+s[1]]].reshape((2, -1)).T
        k = np.r_[ll(0, 0), ll(1, 0), ll(1, 1), ll(0, 1)].reshape((4, -1, 2)).transpose(1, 0, 2)
    else:
        def ll(i, j):
            return np.r_[grid_lon.isel_points(south_north_stag=i, west_east_stag=j),
                         grid_lat.isel_points(south_north_stag=i, west_east_stag=j)].reshape((2,-1)).T
        i, j = np.where(mask)
        k = np.r_[ll(i, j), ll(i+1, j), ll(i+1, j+1), ll(i, j+1)].reshape((4, -1, 2)).transpose(1,0,2)
    lr = [Polygon(LinearRing(a)) for a in k]
    mp = MultiPoint(list(zip(lon, lat)))
    c = [[l for l, r in enumerate(lr) if r.contains(p)] for p in mp]
    l = np.array([len(r)==0 for r in c])
    c = [r[0] for r in c if len(r)>0]
    if mask is None:
        i, j = np.unravel_index(c, s)
        return (i, j, l)
    else:
        return (i[c], j[c], l)
