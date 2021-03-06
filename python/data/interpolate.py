"""
Interpolation routines
----------------------

See https://en.wikipedia.org/wiki/Bilinear_interpolation.
The points in ij, k in :meth:`.weights` are ordered::

       2           3
    x1  x-X1------x
        | |       |
        | |       |
        | o <-ipp |
        | |       |
    x0  x-X0------x
       0           1
      y0          y1
    ^
    | y
        x ->

*ipp* denotes the interpolation point. Four points (0 - 3, in the comments denoted p0 - p4) are selected by computing the distances of *ipp* to each point in the domain, and then summing the distances for all possible 'squares' of four points that can be formed; the one with the lowest sum of 4 distances is the square containing *ipp*.

In the x-direction, two interpolations are performed on the lines denoted *x0* and *x1* (to the points *X0* and *X1*), whereas in the y-directions, the interpolation weights are also computed for the lines *y0* and *y0*, and the final interpolation between *X0* and *X1* is performed with the mean of these two sets of points.


Usage
=====
All classes (:class:`GridInterpolator`, :class:`BilinearInterpolator`) inherit from a the :class:`Interpolator` base class, all of whose keyword arguments can be set on the subclasses too.

Interpolation is carried out by calling either of the two methods :meth:`xarray` or :meth:`netcdf` on the instantiated class, depending on whether the dataset being interpolated is an :class:`xarray.Dataset` or a `netCDF4`_ dataset::

    import xarray as xr

    with xr.open_dataset(...) as ds:
        BLI = BilinearInterpolator(ds, ...)
        result = BLI.xarray(ds['T2'])

If several variables should be interpolated at the same time, one should (probably) use :meth:`~xarray.Dataset.to_array` - also, **this only works as expected if all the variables share the same dimensions** and if ``ds`` is an :class:`xarray.Dataset` and not a `netCDF4`_ one::

    result = BLI.xarray(ds[['T2', 'Q2']].to_array())

On the other hand, it shouldn't be too much of a penaltiy efficiency-wise to apply the interpolation by variable::

    result = ds[['T', 'PH']].apply(BLI.xarray)

"""

import xarray as xr
import numpy as np
import pandas as pd
import unittest, os
from pyproj import Proj
from geo import proj_params
from importlib import import_module
# from timeit import default_timer as timer


def g2d(v):
    m, n = v.shape[-2:]
    return np.array(v[:]).flatten()[-m * n:].reshape((m, n))

class Interpolator:
    """Base class for the interpolators in this module. The parameters common to all routines are described here.

    :param ds: netCDF Dataset from which to interpolate (and which contains projection parameters as attributes).
    :type ds: :class:`xarray.Dataset` or `netCDF4`_ Dataset.

    :Keyword arguments:
        * **stations** - DataFrame with station locations as returned by a call to :meth:`.CEAZA.CEAZAMet.get_stations`.
        * **lon** - :obj:`iterable` of longitudes
        * **lat** - :obj:`iterable` of latitudes
        * **names** - :obj:`iterable` of station names

    .. NOTE::
        None of the keyword arguments are necessary; if none are given, the default list of stations is loaded according to the values specified in the config file (see :mod:`data`). Alternatively, ``lon``, ``lat`` and ``names`` can be specified directly (but ``names``) is currently not used if ``ds`` is a `netCDF4`_ Dataset.

    """
    spatial_dims = ['south_north', 'west_east']
    """The names of the latitude / longitude dimensions"""

    spatial_vars = [['XLONG', 'XLAT'], ['XLONG_M', 'XLAT_M']]

    time_dim_coord = 'XTIME'

    def __init__(self, ds, *args, stations=None, lon=None, lat=None, names=None, **kwargs):
        proj = Proj(**proj_params(ds))
        for V in self.spatial_vars:
            try:
                self.x, self.y = proj(*[g2d(ds[v]) for v in V])
                break
            except: pass
        assert hasattr(self, 'x'), "No horizontal coordinates found."
        if lon is not None:
            self.index = names
            self.ij = proj(lon, lat)
        else:
            if (stations is None):
                stations = pd.read_hdf(self.config.Meta.file_name, 'stations')
            self.index = stations.index
            self.ij = proj(*stations[['lon', 'lat']].values.T)

        # self.start_time = timer()

    def netcdf_dims(self, var):
        dims = var.dimensions
        other_dims = [d for d in dims if d not in self.spatial_dims]
        j = [dims.index(d) for d in self.spatial_dims]
        k = [dims.index(d) for d in other_dims]
        x = var[:].transpose(np.r_[j, k])
        return x, x.shape[:len(j)], x.shape[len(j):], other_dims

class GridInterpolator(Interpolator):
    """Uses :mod:`scipy.interpolate` to interpolate a model field horizontally to station locations. Loops over the non-lon/lat dimensions. Once per instantiation, the following steps are performed:

        #. Project model ``XLONG``/``XLAT`` coordinates according to the grid parameters given as netCDF attributes in the :class:`xarray.Dataset`.
        #. Compute an affine transformation (see :meth:`geo.affine`) from a possibly rotate projected grid to one spanned by simply integer vectors in both directions.

    A call to the returned class instance uses :meth:`scipy.interpolate.interpn` to interpolate from this regular grid to station locations.

    See :class:`InterpolatorBase` for a description of the parameters common to all interpolators.

    :Keyword arguments:
        * **method** - Maps directly to the param of the same name in :func:`~scipy.interpolate.interpn`.

    Interpolation is carried out by calling the instantiated class as described for :class:`.BilinearInterpolator`.
    """
    def __init__(self, ds, *args, method='linear', **kwargs):
        super().__init__(ds, *args, **kwargs)
        from geo import affine
        self.intp = import_module('scipy.interpolate')
        self.method = method
        tg = affine(self.x, self.y)
        self.coords = np.roll(tg(np.r_['0,2', self.ij]).T, 1, 1)
        self.mn = [range(s) for s in self.x.shape]

    def _grid_interp(self, data):
        return np.array([self.intp.interpn(self.mn, data[:, :, k], self.coords, self.method, bounds_error=False)
             for k in range(data.shape[2])])

    def xarray(self, x):
        dims = [d for d in x.dims if d not in self.spatial_dims]
        if len(dims) > 0:
            X = x.stack(n = dims).transpose(*self.spatial_dims, 'n')
            y = self._grid_interp(X.values)
            ds = xr.DataArray(y, coords=[('n', X.indexes['n']), ('station', self.index)])
            if len(dims) > 1:
                ds = ds.unstack('n')
            else:
                # this hack is necessary because I use 'stack' above even if there is only 1 'dims'
                ds.coords['n'] = ('n', ds.indexes['n'].get_level_values(0))
                ds = ds.rename({'n': dims[0]})
            if hasattr(x, self.time_dim_coord):
                xt = x[self.time_dim_coord]
                ds.coords[self.time_dim_coord] = (xt.dims, xt)
        else:
            y = self.intp.interpn(self.mn, x.values, self.coords, self.method, bounds_error=False)
            ds = xr.DataArray(y, coords=[('station', self.index)])
        # print('Time taken: %s', timer() - self.start_time)
        return ds

    def netcdf(self, x):
        if len(x.shape) > 2:
            x, s, r, other_dims = self.netcdf_dims(x)
            x = x.reshape(np.r_[s, [-1]])
            y = (np.moveaxis(np.array(self._grid_interp(x)).reshape(np.r_[r, [-1]]), -1, 0),
                np.r_[['station'], other_dims])
        else:
            y = self.intp.interpn(self.mn, x, self.coords, self.method, bounds_error=False)
        # print('Time taken: %s', timer() - self.start_time)
        return y

class BilinearInterpolator(Interpolator):
    """Implements bilinear interpolation in two spatial dimensions by precomputing a weight matrix which can be used repeatedly to interpolate to the same spatial locations. The input :class:`xarray.Dataset` / `netCDF4`_ Dataset is reshaped into a two-dimensional matrix, with one dimension consisting of the stacked two spatial dimensions (longitude / latitude), and the other dimension consisting of all other dimensions stacked. Hence, interpolation to fixed locations in the horizontal (longitude / latitude) plane can be carried out by a simple matrix multiplication with the precomputed weights matrix. Different model levels, variables and other potential dimensions can be stacked without the need to write loops in python. The interpolation weights are computed in **projected** coordinates.

    See :class:`InterpolatorBase` for a description of the parameters common to all interpolators.
    """

    def __init__(self, ds, *args, **kwargs):
        from geo import Squares
        super().__init__(ds, *args, **kwargs)
        self.points = Squares.compute(self.x, self.y, *self.ij)
        K = np.ravel_multi_index(self.points.sel(var='indexes').astype(int).values,
                                 self.x.shape[:2]).reshape((4, -1))

        n = self.ij[0].size
        self.W = np.zeros((n, np.prod(self.x.shape)))
        self.W[range(n), K] = self.points.groupby('station').apply(self._weights).transpose('square', 'station')

    def xarray(self, x):
        dims = set(x.dims).symmetric_difference(self.spatial_dims)
        if len(dims) > 0:
            X = x.stack(s=self.spatial_dims, t=dims)
            y = xr.DataArray(self.W.dot(X), coords=[self.index, X.coords['t']]).unstack('t')
            if hasattr(x, self.time_dim_coord):
                y.coords[self.time_dim_coord] = x.coords[self.time_dim_coord]
        else:
            X = x.stack(s=self.spatial_dims)
            y = xr.DataArray(self.W.dot(X), coords=[self.index])
        # print('Time taken: %s', timer() - self.start_time)
        return y

    # this appears to be returning a tuple of data, dims
    def netcdf(self, x):
        if len(x.shape) <= 2:
            raise Exception('2- or lower-dimensional data unsupported')
        x, s, r, other_dims = self.netcdf_dims(x)
        return (self.W.dot(x.reshape((np.prod(s), -1))).reshape(np.r_[[-1], r]),
                np.r_[['station'], other_dims])

    def plot(self, k):
        import matplotlib.pyplot as plt
        p = self.points.sel(var='points').isel(station=k).transpose('xy', 'square').values
        plt.figure()
        plt.scatter(self.x, self.y, marker='.')
        plt.scatter(*p, marker='o')
        plt.scatter(self.i[k], self.j[k], marker='x')
        plt.show()

    def _weights(self, D):
        # this are the 4 'square' point coordinates
        p = D.sel(var='points').transpose('square', 'xy').values
        DX, DY = D.sel(var='distances').transpose('xy', 'square').values

        # x-direction
        dx0 = p[1] - p[0] # vector p0 -> p1
        dx1 = p[3] - p[2] # vector p2 -> p3
        x0 = np.dot(dx0, dx0) ** .5 # dx0 magnitude
        x1 = np.dot(dx1, dx1) ** .5 # dx1 magnitude

        # this computes orthogonal projections of the vectors pointing from each of the 4 'square'
        # to the interpolation point (ipp) onto the vectors between points (dx0 and dx1)
        a = np.dot(dx0, [DX[0], DY[0]]) / x0
        b = np.dot(dx1, [DX[2], DY[2]]) / x1
        c = np.dot(dx0, [DX[1], DY[1]]) / x0
        d = np.dot(dx1, [DX[3], DY[3]]) / x1

        # projections of the vectors p0 -> ipp and p2 -> ipp need to be positive,
        # of p1 -> ipp and p3 -> ipp negative, otherwise the point is outside the domain
        if (a < 0) or (b < 0) or (c > 0) or (d > 0):
            return xr.DataArray(np.zeros(4) * np.nan, dims=['square'])

        # we compute two interpolates along the lines 'x0' and 'x1' ('X0' and 'X1' on drawing),
        # one between p0 and p1 and one between p2 and p3
        w1 = [-c, a] / x0
        w2 = [-d, b] / x1

        # y-direction
        dy0 = p[2] - p[0]
        dy1 = p[3] - p[1]
        y0 = np.dot(dy0, dy0) ** .5
        y1 = np.dot(dy1, dy1) ** .5

        a = np.dot(dy0, [DX[0], DY[0]]) / y0
        b = np.dot(dy1, [DX[1], DY[1]]) / y1
        c = np.dot(dy0, [DX[2], DY[2]]) / y0
        d = np.dot(dy1, [DX[3], DY[3]]) / y1

        if (a < 0) or (b < 0) or (c > 0) or (d > 0):
            return xr.DataArray(np.zeros(4) * np.nan, dims=['square'])

        # we use the mean of the two sets of weights computed for the y-direction (along 'y0' and 'y1')
        wy = np.vstack(([-c, a] / y0, [-d, b] / y1)).mean(0)
        return xr.DataArray(np.r_[wy[0] * w1, wy[1] * w2], dims=['square'])

# NOTE: I don't think this data is still around, use the tests in WRF.py
class Test(unittest.TestCase):
    def setUp(self):
        d = '/nfs/temp/sata1_ceazalabs/carlo/WRFOUT_OPERACIONAL/c01_2015051900/wrfout_d03*'
        self.ds = xr.open_mfdataset(d)
        sta = pd.read_hdf(self.station_meta, 'stations')
        self.intp = BilinearInterpolator(self.ds, sta)

    def tearDown(self):
        self.ds.close()

    def test_1(self):
        x = self.intp(self.ds['T2'].sortby('XTIME'))
        with xr.open_dataarray('/home/arno/Documents/code/python/Bilinear_test.nc') as y:
            np.testing.assert_allclose(x, y)


if __name__=='__main__':
    unittest.main()
