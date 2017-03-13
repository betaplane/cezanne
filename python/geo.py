#!/usr/bin/env python
import gdal
import pandas as pd
import numpy as np
from mpl_toolkits import basemap


class angle(object):
    def __init__(self, gdalDS):
        self._geo = gdalDS.GetGeoTransform()
        self.map = basemap.Basemap(
            projection = 'merc',
            llcrnrlon = self._geo[0],
            llcrnrlat = self._geo[3] - ds.RasterYSize * self._geo[5],
            urcrnrlon = self._geo[0] + ds.RasterXSize * self._geo[1],
            urcrnrlat = self._geo[3]
            )
        self._i = np.arange(gdalDS.RasterXSize, dtype=np.int)
        self._j = np.arange(gdalDS.RasterYSize, dtype=np.int)
        self.z = ds.GetRasterBand(1).ReadAsArray()


    def set_station(self, station):
        self._x = int((station.lon - self._geo[0]) // self._geo[1])
        self._y = int((station.lat - self._geo[3]) // self._geo[5])
        # subtract station elevation from grid
        # first dim of matrix is lat, second lon
        z = self.z[self._y, self._x]
        self.z -= z
        print('elevation difference station - grid: {:.0f} m'.format(station.elev - z))
        lon, lat = np.meshgrid(self._i + .5, self._j + .5)
        # self._geo[5] is already negative
        x, y = self.map(self._geo[0] + lon * self._geo[1], self._geo[3] + lat * self._geo[5])
        sx, sy = self.map(station.lon, station.lat)
        self.dist = ((x - sx)**2 + (y - sy)**2)**.5
        self.angle = np.arctan(self.z / self.dist)

    def window(self, dx, x, y=None):
        try:
            if len(x.shape) > 1:
                return x[self._y-dx: self._y+dx, self._x-dx: self._x+dx]
            else:
                return x[self._x-dx: self._x+dx]
                return y[self._y-dx: self._y+dx]
        except:
            pass
        try:
            return x - self._x + dx, y - self._y + dx
        except:
            pass

    # line drawing algorithm
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    def path(self, a):
        """
        Compute a path in pixel coordinates going through the station location (self._x, self._y)
        at angle 'a' from east (normal math convention).
        :param a: Angle from east in radians.
        :returns: i, j vectors containing the pixel coordinates.
        """
        sa = np.sin(a)
        ca = np.cos(a)
        if ca == 0:
            i = np.ones(len(self._j), dtype=np.int) * self._x
            return i, self._j
        if sa == 0:
            j = np.ones(len(self._i), dtype=np.int) * self._y
            return self._i, j
        else:
            ta = abs(np.tan(a))
            # remember y is positive south
            # x > y
            if ta < 1:
                j = np.round(-np.sign(sa) * ta * self._i).astype(int)
                # shift so that line goes through station pixel
                j += self._y - j[self._x]
                ix = (j >= 0) & (j <= self._j[-1])
                return self._i[ix], j[ix]
            # y > x
            else:
                i = np.round(-np.sign(ca) * abs(ca / sa) * self._j).astype(int)
                i += self._x - i[self._y]
                ix = (i >= 0) & (i <= self._i[-1])
                return i[ix], self._j[ix]

    def max_ang(self, ang, dn):
        i, j = self.path(ang)
        # remember i, j reversed
        d = self.dist[j, i]
        n = np.argmin(d)
        a = self.angle[j, i]
        m1 = np.argmax(a[n+1+dn:])
        m2 = np.argmax(a[:n-dn])
        return (a[n+1+m1], a[m2], d[n+1+dn+m1], d[m2]) if ang < np.pi/8 else (a[m2], a[n+1+m1], d[m2], d[n+1+dn+m1])

    def circle(self, res, from_north=True, station=None, dx=2):
        """
        Compute the elevation angle of mountains surrounding a station in a circle starting from north,
        counter-clockwise. Simple planar geometry is used, no atmospheric refraction.
        :param res: Resolution of full circle (360 -> 1 deg)
        :param from_north: If set to False, use the usual mathematical convention and start circle in east.
        :param station: If, given, set the station first.
        :param dx: Number of pixels to exclude immediately surrounding the station. Leads to less grid artefacts.
        :returns: DataFrame with radial angles (from north/east) as index and columns 'alt' (elevation angle)
        and 'dist' (distance from station at which maximum elevation angle is found).
        """
        if station is not None:
            print('Setting station {}'.format(station.name))
            try:
                self.set_station(station)
            except:
                print('Station {} outside map.'.format(station.name))
                return pd.DataFrame()
            else:
                print('Setting station {}.'.format(station.name))

        da = 2*np.pi / res * np.arange(res/2)
        c = np.array([self.max_ang(a, dx) for a in da]).T.flatten()
        return pd.DataFrame(c.reshape((2, res)).T, columns=['alt', 'dist'], index=np.r_[da,da+np.pi])


if __name__ == "__main__":
    ds = gdal.Open('../../data/geo/merged.tif')
    D = pd.HDFStore('../../data/tables/station_data_new.h5', 'r')
    sta = D['sta']
    st = sta.loc['5']

    A = angle(ds)

    P = pd.Panel(dict([(c, A.circle(360, station=st)) for c, st in sta.iterrows()]))
    R = pd.HDFStore('../../data/tables/radiation.h5')
    R['shading'] = P
    R.close()
