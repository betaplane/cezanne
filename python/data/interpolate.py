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

.. NOTE::

    * For now, import troubles (can't import from higher level than package) are circumvented with sys.path.append().

"""

import sys
sys.path.append('..')
import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate as ip
from pyproj import Proj
from geo import proj_params, affine
import helpers as hh
import unittest


class GridInterpolator(object):
    """Uses :mod:`scipy.interpolate` to interpolate a model field horizontally to station locations. Loops over the non-lon/lat dimensions. Once per instantiation, the following steps are performed:

        #. Project model ``XLONG``/``XLAT`` coordinates according to the grid parameters given as netCDF attributes in the :class:`xarray.Dataset`.
        #. Compute an affine transformation (see :meth:`geo.affine`) from a possibly rotate projected grid to one spanned by simply integer vectors in both directions.

    A call to the returned class instance uses :meth:`scipy.interpolate.interpn` to interpolate from this regular grid to station locations.

    :param ds: netCDF Dataset from which to interpolate (and which contains projection parameters as attributes).
    :type ds: :class:`~xarray.Dataset`
    :param stations: DataFrame with station locations as returned by a call to :meth:`.CEAZA.Downloader.get_stations`.
    :type stations: :class:`~pandas.DataFrame`
    :param method: Maps directly to the param of the same name in :func:`~scipy.interpolate.interpn`.

    """
    def __init__(self, ds, stations, method='linear'):
        self.method = method
        proj = Proj(**proj_params(ds))
        xy = proj(hh.g2d(ds['XLONG']), hh.g2d(ds['XLAT']))
        tg = affine(*xy)
        ij = proj(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
        self.coords = np.roll(tg(np.r_['0,2', ij]).T, 1, 1)
        self.mn = (range(xy[0].shape[0]), range(xy[0].shape[1]))
        self.index = stations.index

    def _grid_interp(self, data):
        return [ip.interpn(self.mn, data[:, :, k], self.coords, self.method, bounds_error=False)
             for k in range(data.shape[2])]

    def __call__(self, v):
        x = v.stack(n = v.dims[:-2]).transpose(*v.dims[-2:], 'n')
        y = self._grid_interp(x.values)
        ds = xr.DataArray(y, coords=[('n', x.indexes['n']), ('station', self.index)]).unstack('n')
        ds.coords['XTIME'] = ('Time', v.XTIME)
        return ds

class BilinearInterpolator(object):
    def __init__(self, ds, stations):
        pr = Proj(**proj_params(ds))
        self.x, self.y = pr(hh.g2d(ds.XLONG), hh.g2d(ds.XLAT))
        self.index = stations.index
        self.i, self.j = pr(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
        self.dx = self.i.reshape((1, 1, -1)) - np.expand_dims(self.x, 2)
        self.dy = self.j.reshape((1, 1, -1)) - np.expand_dims(self.y, 2)

        D = (self.dx**2 + self.dy**2) ** .5
        # this is the sum over all distances of four points arranged in a square
        self.D = D[:-1, :-1, :] + D[1:, :-1, :] + D[:-1, 1:, :] + D[1:, 1:, :]

        self.outliers = []
        self.W = np.zeros((stations.shape[0], np.prod(self.x.shape)))
        for k in range(stations.shape[0]):
            self._weights(k)

    def __call__(self, x):
        X = x.stack(s=x.dims[-2:], t=x.dims[:-2])
        y =  xr.DataArray(self.W.dot(X), coords=[self.index, X.coords['t']]).unstack('t')
        y.coords['XTIME'] = x.coords['XTIME']
        return y

    def plot(self, k):
        import matplotlib.pyplot as plt
        i, j = np.unravel_index(self.D[:, :, r].argmin(), self.D.shape[:2])
        ij = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
        p = np.array([(self.x[j], self.y[j]) for j in ij])
        plt.figure()
        plt.scatter(self.x, self.y, marker='.')
        plt.scatter(*p.T, marker='o')
        plt.scatter(self.i[k], self.j[k], marker='x')
        plt.show()

    def _weights(self, r):
        # this finds the square with the lowest sum of distances to any interpolation point
        i, j = np.unravel_index(self.D[:, :, r].argmin(), self.D.shape[:2])
        # this sorts the points in the sense of the drawing at the beginning of file
        ij = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]

        # this are the 4 'square' point coordinates
        p = np.array([(self.x[j], self.y[j]) for j in ij])

        DX = [self.dx[m, n, r] for m, n in ij]
        DY = [self.dy[m, n, r] for m, n in ij]

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
            self.W[r, :] = np.nan

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
            self.W[r, :] = np.nan

        # we use the mean of the two sets of weights computed for the y-direction (along 'y0' and 'y1')
        wy = np.vstack(([-c, a] / y0, [-d, b] / y1)).mean(0)

        # this is the new order in the flattened arrays
        k = np.ravel_multi_index(list(zip(*ij)), self.x.shape)
        self.W[r, k] = np.r_[wy[0] * w1, wy[1] * w2]

class Test(unittest.TestCase):
    def setUp(self):
        d = '/nfs/temp/sata1_ceazalabs/carlo/WRFOUT_OPERACIONAL/c01_2015051900/wrfout_d03*'
        self.ds = xr.open_mfdataset(d)
        with pd.HDFStore('/home/arno/Documents/data/CEAZAMet/stations.h5') as S:
            self.intp = BilinearInterpolator(self.ds, S['stations'])

    def tearDown(self):
        self.ds.close()

    def test_1(self):
        x = self.intp(self.ds['T2'].sortby('XTIME'))
        with xr.open_dataarray('/home/arno/Documents/code/python/Bilinear_test.nc') as y:
            np.testing.assert_allclose(x, y)


if __name__=='__main__':
    unittest.main()
