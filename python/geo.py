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
            llcrnrlat = self._geo[3] - gdalDS.RasterYSize * self._geo[5],
            urcrnrlon = self._geo[0] + gdalDS.RasterXSize * self._geo[1],
            urcrnrlat = self._geo[3]
            )
        self._i = np.arange(gdalDS.RasterXSize, dtype=np.int)
        self._j = np.arange(gdalDS.RasterYSize, dtype=np.int)
        self.z = gdalDS.GetRasterBand(1).ReadAsArray()
        lon, lat = np.meshgrid(self._i + .5, self._j + .5)
        # self._geo[5] is already negative
        self._xy = self.map(self._geo[0] + lon * self._geo[1], self._geo[3] + lat * self._geo[5])


    def set_station(self, station):
        self.station_coords(station, get=False)
        sx, sy = self.map(station.lon, station.lat)
        try:
            z = self.z[self.sj, self.si]
        except:
            raise Exception('{} outside map boundaries'.format(station.name))
        self.dz = z - station.elev
        print('{}: elevation difference grid - station: {:.0f} m'.format(station.name, self.dz))
        self.dist = ((self._xy[0] - sx)**2 + (self._xy[1] - sy)**2)**.5
        self.angle1 = np.arctan((self.z - z) / self.dist)
        self.angle2 = np.arctan((self.z - station.elev) / self.dist)

    def station_coords(self, station=None, get=True):
        if station is not None:
            self.si = int((station.lon - self._geo[0]) // self._geo[1])
            self.sj = int((station.lat - self._geo[3]) // self._geo[5])
            if (self.si < 0) or (self.sj < 0) or (self.si >= self._i[-1]) or (self.sj >= self._j[-1]):
                raise Exception('{} outside map boundaries'.format(station.name))
        if get:
            return self.sj, self.si


    def window(self, dx, x=None, y=None):
        try:
            if len(x.shape) > 1:
                return x[self.sj-dx: self._y+dx+1, self.si-dx: self.si+dx+1]
            else:
                return x[self.si-dx: self.si+dx+1]
                return y[self.sj-dx: self._y+dx+1]
        except:
            pass

    # line drawing algorithm
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    def path(self, a):
        """
        Compute a path in pixel coordinates going through the station location (self.si, self.sj)
        at angle 'a' from east (normal math convention).
        :param a: Angle from east in radians.
        :returns: i, j vectors containing the pixel coordinates.
        """
        sa = np.sin(a)
        ca = np.cos(a)
        if ca == 0:
            i = np.ones(len(self._j), dtype=np.int) * self.si
            return i, self._j
        if sa == 0:
            j = np.ones(len(self._i), dtype=np.int) * self.sj
            return self._i, j
        else:
            ta = abs(np.tan(a))
            # remember y is positive south
            # x > y
            if ta < 1:
                j = np.round(-np.sign(sa) * ta * self._i).astype(int)
                # shift so that line goes through station pixel
                j += self.sj - j[self.si]
                ix = (j >= 0) & (j <= self._j[-1])
                return self._i[ix], j[ix]
            # y > x
            else:
                i = np.round(-np.sign(ca) * abs(ca / sa) * self._j).astype(int)
                i += self.si - i[self.sj]
                ix = (i >= 0) & (i <= self._i[-1])
                return i[ix], self._j[ix]

    def max_alt(self, ang, dn):
        i, j = self.path(ang)
        d = self.dist[j, i] # remember i, j reversed
        n = np.argmin(d)
        def ang_dist(angle):
            a = angle[j,i]
            m1 = np.argmax(a[n+1+dn:])
            m2 = np.argmax(a[:n-dn])
            if ang < np.pi/8:
                return (a[n+1+dn+m1], a[m2], d[n+1+dn+m1], d[m2])
            else:
                return (a[m2], a[n+1+dn+m1], d[m2], d[n+1+dn+m1])
        return np.array([ang_dist(a) for a in [self.angle1, self.angle2]]).flatten()

    def circle(self, res, station=None, from_north=True, dx=2):
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
            try:
                print('Setting station {}.'.format(station.name))
                self.set_station(station)
            except Exception as excp:
                print(excp)
                return pd.DataFrame()

        da = 2*np.pi / res * np.arange(res/2)
        c = np.array([self.max_alt(a, dx) for a in da]).T.flatten()
        return pd.DataFrame(
            c.reshape((4, res)).T,
            columns=['alt_grid', 'dist_grid', 'alt_st', 'dist_st'],
            index=np.r_[da,da+np.pi]
        )


if __name__ == "__main__":
    # ds = gdal.Open('../../data/geo/merged.tif')
    D = pd.HDFStore('../../data/tables/station_data_new.h5', 'r')
    sta = D['sta']
    st = sta.loc['5']

    # A = angle(ds)
    # c = A.circle(360, st)
    # circles = [(c, A.circle(360, st)) for c, st in sta.iterrows()]
    # P = pd.Panel(dict([a for a in circles]))
    R = pd.HDFStore('../../data/tables/radiation.h5')
    # R['shading'] = P
    # R.close()
    sha = R['shading']


