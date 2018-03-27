"""
The points in ij, k in :meth:`.weights` are ordered::

       2         3
    x1  x-------x
        |       |
        |  o    |
        |       |
    x0  x-------x
       0         1
      y0        y1

"""

import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Proj
from geo import proj_params
import helpers as hh
import unittest

class BilinearInterpolator(object):
    def __init__(self, ds, stations):
        pr = Proj(**proj_params(ds))
        self.x, self.y = pr(hh.g2d(ds.XLONG), hh.g2d(ds.XLAT))
        self.index = stations.index
        self.i, self.j = pr(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
        self.dx = self.i[np.newaxis, :] - self.x.flatten()[:, np.newaxis]
        self.dy = self.j[np.newaxis, :] - self.y.flatten()[:, np.newaxis]
        # self.dx = self.i.reshape((1, -1)) - self.x.reshape(np.r_[np.prod(self.x.shape), 1])

        self.dist = self.dx**2 + self.dy**2

        self.p4 = np.apply_along_axis(np.argsort, 0, self.dist)[:4, :]
        self.outliers = []
        self.W = np.vstack([self.weights(k) for k in range(stations.shape[0])])

    def interpolate(self, x):
        X = x.stack(s=x.dims[-2:], t=x.dims[:-2])
        y =  xr.DataArray(self.W.dot(X), coords=[self.index, X.coords['t']])
        return y.unstack('t')

    def plot(self, k):
        import matplotlib.pyplot as plt
        dd = self.dist[:, k].reshape(self.x.shape)
        D = dd[:-1, :-1] + dd[1:, :-1] + dd[:-1, 1:] + dd[1:, 1:]
        i, j = np.unravel_index(D.argmin(), D.shape)
        ij = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
        p = np.array([(self.x[j], self.y[j]) for j in ij])
        plt.figure()
        plt.scatter(self.x, self.y, marker='.')
        plt.scatter(*p.T, marker='o')
        plt.scatter(self.i[k], self.j[k], marker='x')
        plt.show()

    def weights(self, i):
        # this sorts the points by first, second index (see drawing in beginning of file)
        ij = sorted(zip(*np.unravel_index(self.p4[:, i], self.x.shape)))

        # this are the 4 point coordinates
        p = np.array([(self.x[j], self.y[j]) for j in ij])

        # this is the new order in the flattened arrays
        k = np.ravel_multi_index(list(zip(*ij)), self.x.shape)
        DX = self.dx[k, i]
        DY = self.dy[k, i]

        w = np.zeros(self.dx.shape[0])

        # x-direction
        dx0 = p[1] - p[0]
        dx1 = p[3] - p[2]
        x0 = np.dot(dx0, dx0) ** .5
        x1 = np.dot(dx1, dx1) ** .5

        a = np.dot(dx0, [DX[0], DY[0]]) / x0
        b = np.dot(dx1, [DX[2], DY[2]]) / x1
        c = np.dot(dx0, [DX[1], DY[1]]) / x0
        d = np.dot(dx1, [DX[3], DY[3]]) / x1

        test = (a > 0) and (b > 0) and (c < 0) and (d < 0)
        if not test:
            self.outliers.append(i)

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

        test = (a > 0) and (b > 0) and (c < 0) and (d < 0)
        if not test:
            self.outliers.append(i)

        wy = np.vstack(([-c, a] / y0, [-d, b] / y1)).mean(0)
        w[k] = np.r_[wy[0] * w1, wy[1] * w2]
        return w

class Test(unittest.TestCase):
    def setUp(self):
        d = '/nfs/temp/sata1_ceazalabs/carlo/WRFOUT_OPERACIONAL/c01_2015051900/wrfout_d03*'
        self.ds = xr.open_mfdataset(d)
        with pd.HDFStore('/home/arno/Documents/data/CEAZAMet/stations.h5') as S:
            self.intp = BilinearInterpolator(self.ds, S['stations'])

    def tearDown(self):
        self.ds.close()

    def test_1(self):
        x = self.intp.interpolate(self.ds['T2'].sortby('XTIME'))
        with xr.open_dataarray('/home/arno/Documents/code/python/data/Bilinear_test.nc') as y:
            np.testing.assert_allclose(x, y)


if __name__=='__main__':
    unittest.main()
