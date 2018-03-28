#!/usr/bin/env python
import numpy as np
from xarray import DataArray, open_dataset, concat
from functools import singledispatch, partial
from cartopy.io import shapereader

def kml(name, lon, lat, code=None, nc=None):
    from simplekml import Kml, Style
    from shapely import Polygon, Point
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

def df2kml(df, name, body=None, lon='lon', lat='lat'):
    """Generate a :class:`~simplekml.Kml` object from a :class:`~pandas.DataFrame`. Save to file by calling :meth:`~simplekml.Kml.save` on the returned :class:`~simplekml.Kml` object. Example usage::

        kml = df2kml(df, ('{} ({})', ['id', 'name']),
                     body = ('From {} to {}', ['mindate', 'maxdate']),
                     lon='longitude', lat='latitude')
        kml.save('example.kml')

    :param df: DataFrame containing the data from which to generate the :class:`~simplekml.Kml` object.
    :type df: :class:`~pandas.DataFrame`
    :param name: Tuple of the form (``format``, ``columns``) where format is a format string to be filled by the items in ``columns``. ``Columns`` should always be given as a list.
    :type name: :obj:`tuple` or :obj:`list`
    :param body: Same as for ``name``, but for the popup ballon text.
    :type body: :obj:`tuple` or :obj:`list`
    :param lon: Name of the column containing the longitude data.
    :param lat: Name of the column containing the latitude data.
    :rtype: :obj:`~simplekml.Kml` object

    """
    from simplekml import Kml, Style
    k = Kml()
    for _, s in df.iterrows():
        p = k.newpoint(name=name[0].format(*s[name[1]]), coords=[s[[lon, lat]]])
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        if body is not None:
            p.style.balloonstyle.text = body[0].format(*s[body[1]])
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
    from shapely import MultiPoint, Polygon, LinearRing
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
    c = [r[0] for r in c if len(r)>0]
    return np.unravel_index(c, s) if mask is None else (i[c], j[c])

class Squares(object):
    """Determine the 4 points in a model grid which form a rectangular grid cell containing any points of interest, for example for interpolation (see :mod:`.data.interpolate`). All methods return a DataArray containing:

            * projected coordinates (*x* and *y*) of the four 'square' points for each point of interest (dimension ``var='points'``)
            * *x* and *y* distances of the four 'square' points from POI (``var='distances'``)
            * indexes of the four points in the shape of the ``x`` and ``y`` variables (``var='indexes'``)

The *x* and *y* components of each of these ``var`` dimensions are accessed via a dimension called ``xy``.

    """
    @classmethod
    def project(cls, ds, stations):
        """Project coordinates from a netCDF file and station longitude / latitude data first, then call :meth:`.compute`.

        :param ds: Dataset containing the coordinate variables and projection paramters
        :type ds: :class:`~xarray.Dataset`
        :param stations: DataFrame containing the locations (longitude / latitude) for which the surrounding 4 points should be determined (as returned from :meth:`.CEAZA.Downloader.get_stations`)
        :type stations: :class:`~pandas.DataFrame`
        :returns: see :class:`.Squares` for description of returned DataArray
        :rtype: :class:`~xarray.DataArray`

        """
        from helpers import g2d
        from pyproj import Proj

        pr = Proj(**proj_params(ds))
        x, y = pr(g2d(ds.XLONG), g2d(ds.XLAT))
        k, l = pr(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
        return cls.compute(x, y, k, l)

    @staticmethod
    def compute(x, y, k, l):
        """Compute the actual containing 'squares' surrounding a point of interest (POI) from projected coordinates.

        :param lon: 2-dimensional (matrix-ordered) projected x (longitude) coordinates
        :param lat: 2-dimensional (matrix-ordered) projected y (latitude) coordinates
        :param k: 1-dimensional array of projected x-direction coordinates of the POI
        :param l: 1-dimensional array of projected y-direction coordinates of the POI
        :returns: see :class:`.Squares` for description of returned DataArray
        :rtype: :class:`~xarray.DataArray`

        """
        from pandas import Index
        dx = k.reshape((1, 1, -1)) - np.expand_dims(lon, 2)
        dy = l.reshape((1, 1, -1)) - np.expand_dims(lat, 2)

        d = (dx**2 + dy**2) ** .5
        # this is the sum over all distances of four points arranged in a square
        D = d[:-1, :-1, :] + d[1:, :-1, :] + d[:-1, 1:, :] + d[1:, 1:, :]
        i, j = np.unravel_index(D.reshape((-1, D.shape[-1])).argmin(0), D.shape[:2])

        n = np.arange(k.shape[0])
        x = DataArray(x[[i, i, i+1, i+1], [j, j+1, j, j+1]], dims=['square', 'station'])
        y = DataArray(y[[i, i, i+1, i+1], [j, j+1, j, j+1]], dims=['square', 'station'])
        DX = DataArray(dx[[i, i, i+1, i+1], [j, j+1, j, j+1], n], dims=['square', 'station'])
        DY = DataArray(dy[[i, i, i+1, i+1], [j, j+1, j, j+1], n], dims=['square', 'station'])
        I = DataArray([i, i, i+1, i+1], dims=['square', 'station'])
        J = DataArray([j, j+1, j, j+1], dims=['square', 'station'])

        points = concat((x, y), Index(['x', 'y'], name='xy'))
        dists = concat((DX, DY), Index(['x', 'y'], name='xy'))
        indexes = concat((I, J), Index(['x', 'y'], name='xy'))
        return concat((points, dists, indexes), Index(['points', 'distances', 'indexes'], name='var'))



def nearest(grid_lon, grid_lat, lon, lat):
    from pyproj import Geod
    inv = partial(Geod(ellps='WGS84').inv, lon, lat)

    def dist(x, y):
        return inv(x, y)[2]

    dv = np.vectorize(dist)
    d = dv(grid_lon, grid_lat)
    lat_idx, lon_idx = np.unravel_index(np.argmin(d), grid_lon.shape)
    return lat_idx, lon_idx, d[lat_idx, lon_idx]

def distance_matrix(grid_lon, grid_lat, lon, lat):
    """Given grids of longitudes and latitudes, return a matrix with the distances from a point.

    :param grid_lon: grid of longitudes
    :param grid_lat: grid of latitudes
    :param lon: longitude of point from which distances should be computed
    :param lat: latitude of point
    :returns: distance matrix of same shape as grid_lon and grid_lat
    :rtype: :class:`np.array`

    """
    from pyproj import Geod
    inv = partial(Geod(ellps='WGS84').inv, lon, lat)
    def dist(x, y):
        return inv(x, y)[2]
    return np.vectorize(dist)(grid_lon, grid_lat)

@singledispatch
def domain_bounds(ds, project=True, test=None):
    from shapely.geometry import MultiPoint, Polygon, LinearRing
    if project:
        from pyproj import Proj
        pr = Proj(**proj_params(ds))
        coords = np.vstack(pr(ds.corner_lons[-4:], ds.corner_lats[-4:])).T
        if test is not None:
            test = np.vstack(pr(*test.T)).T
    else:
        coords = np.vstack((ds.corner_lons[-4:], ds.corner_lats[-4:])).T
    p = Polygon(LinearRing(coords))
    if test is None:
        return p
    else:
        return [p.contains(i) for i in MultiPoint(test)]

@domain_bounds.register(str)
def _(fn, test=None):
    with open_dataset(fn) as ds:
        return domain_bounds(ds, test)

def cartopy_params(params):
    return {
        'central_longitude': params['lon_0'],
        'central_latitude':  params['lat_0'],
        'standard_parallels':  (params['lat_1'], params['lat_2']),
    }

def proj_params(file, proj='lcc'):
    return {
        'lon_0': file.CEN_LON,
        'lat_0': file.CEN_LAT,
        'lat_1': file.TRUELAT1,
        'lat_2': file.TRUELAT2,
        'proj': proj
    }


@singledispatch
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

@affine.register(DataArray)
def _(x, y):
    return affine(x.values, y.values)


class box(object):
    def __init__(self, *x):
        self.x1, self.x2 = sorted(x[:2])
        self.y1, self.y2 = sorted(x[2:])

    @property
    def rect(self):
        return [(self.x1, self.x1, self.x2, self.x2, self.x1), (self.y1, self.y2, self.y2, self.y1, self.y1)]

    def isel(self, x):
        i = np.where((x.lon >= min(self.x1, self.x2)) & (x.lon <= max(self.x1, self.x2)))[0]
        j = np.where((x.lat >= min(self.y1, self.y2)) & (x.lat <= max(self.y1, self.y2)))[0]
        return x.isel(lon=i, lat=j)

    def inside(self, lon, lat):
        return (self.x1 <= lon <= self.x2) & (self.y1 <= lat <= self.y2)

    def stations(self, stations, lon, lat):
        sta = stations[[lon, lat]]
        return sta.loc[[self.inside(*st) for st in sta.as_matrix()]]


class GeoBase(object):
    """
A class to hold transformations from two Latitude/Longitue dimensions on a :class:`xarray.DataArray` to a single dimension (e.g. for regression purposes).
    """
    def __init__(self, Y, mask=None, mean=False, std=False):
        s = lambda x: x.stack(space = ('lat', 'lon'))
        if mean:
            Y = Y - Y.mean('time')
        if std:
            Y = Y / Y.std('time')
        self.Y = Y.stack(space = ('lat', 'lon')).transpose('time', 'space')
        if mask is not None: # check if this is even necessary if masked values are nan and stack() is used
            self.Y = self.Y.sel(space = s(mask).values.astype(bool).flatten())
        self.N, self.D = self.Y.shape


class sshfs_shapereader(shapereader.Reader):
    def __init__(self, path, sshfs):
        from shapefile import Reader
        self._sshfs = {k: sshfs.openbin('.'.join((path, k))) for k in ['shp', 'shx', 'dbf']}
        self._reader = Reader(**self._sshfs)
        self._geometry_factory = shapereader.GEOMETRY_FACTORIES.get(self._reader.shapeType)
        self._fields = self._reader.fields

    def __del__(self):
        for f in self._sshfs.values():
            f.close()
