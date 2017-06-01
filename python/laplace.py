#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg as la
import helpers as hh
import matplotlib.pyplot as plt




# This is an implementation of a linear regression local in space and time (but over some neighborhood), with
# spatial smoothness of the regression coefficients enforced by regularization via a spatial graph Laplacian.
# Based on:
# Subbian, Karthik, and Arindam Banerjee. “Climate Multi-Model Regression Using Spatial Smoothing.”
# In SDM, 324–332. SIAM, 2013. http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.36.

class GLR(object):
    def __init__(self, distances, weights=None, laplacian=None):
        d = xr.DataArray(distances, dims=('space', 'i'))
        if weights is None:
            self.W = 1 - d / d.max()
        else:
            self.W = weights(d)
        if laplacian is None:
            self.L = self.W
        else:
            self.L = laplacian(d)

    def time_laplacian(self, t=None, size=None, func=None):
        if size is None:
            DT = np.abs([[i - j for i in t] for j in t])
        else:
            DT = 1 - la.toeplitz(np.linspace(0, 1, size))
        if func is not None:
            DT = func(DT)
        K = np.kron(DT, self.L)
        i = pd.MultiIndex.from_product(([0, 1], [0, 1]))
        self.K = xr.DataArray(np.diag(np.array(K.sum(1))) - K, coords=[i, i])

    def _block(self, X, Y):
        x, y = xr.broadcast(X * self.W, Y * self.W, exclude=['var'])

        # instead of fitting intercept (not desirable for regularized regression)
        # remove mean from predictors, estimate intercept as mean of target
        # (see Hastie et al.)
        xm = (x.sum('space') / self.W.sum('space')).mean('time')
        ym = (y.sum('space') / self.W.sum('space')).mean('time')

        x = x - xm * self.W # important!
        y = y - ym * self.W # mean needs to be multiplied by mask
        C = (x * y).sum(('time', 'space')).transpose('i', 'var')
        a = x.transpose('i','var','space','time')
        A = la.block_diag(*np.einsum('ijkl,imkl->ijm', a, a))
        return A, C, xm, ym

    def regress(self, X, Y, lda, window=None, step=None, laplacian=None):
        """
        :param X: predictor
        :type X: xarray with dims 'var', 'space', 'time' (NO intercept!)
        :param Y: target
        :type Y: xarray with dims 'space', 'time'
        :param lda: regularization parameter (lambda)
        :param window: window width for weighted regression in time (in array indexes)
        :param step: increment by which the window to shift (in array indexes)
        :param laplacian: set if the time Laplacian should be computed from the data's time axis
        :type laplacian: callable
        """
        nvar = X['var'].size

        if laplacian is not None:
            self.time_laplacian(Y['time'], func=laplacian)

        if hasattr(self, 'K'):
            # in case we have a space-time Laplacian
            i = pd.MultiIndex.from_product(([0, 1], X['space']))
            L = self.K.loc[i, i]
        else:
            # in case we only have a space Laplacian
            i = X['space']
            L = self.L.loc[i, i]

        # we prepare one time 'chunk' at a time - we need all the chunks eventually for
        # the linear system, but the subroutine only takes as much as needed as per time weighting
        # - but it does deal with all space at a time, both for weighting and arranging it for
        # the graph Laplacian
        A, C, xm, ym = zip(*[self._block(
            X.isel(time=slice(k, k+window)),
            Y.isel(time=slice(k, k+window))
        ) for k in range(0, Y['time'].size, step)])

        L = np.kron(L, np.identity(nvar))
        A = la.block_diag(*A)
        C = np.vstack(C).flatten()
        p = la.solve(A + lda * L, C).reshape((-1, nvar)) # i x nvar

        # the intercept is computed from mean of y plus mean x dot beta
        icpt = np.vstack(ym).reshape((-1, 1)) - np.sum(np.vstack(xm) * p, 1).reshape((-1,1))
        p = np.r_['1,2', p, icpt]
        coords = [i, np.r_[X['var'], ['icpt']]]
        # coords = [i, X['var']]
        P = xr.DataArray(p, coords=coords, dims=['i', 'var']).unstack('i')
        self.p = P.rename({'i_level_0': 'time', 'i_level_1': 'space'})

    @staticmethod
    def slicer(X, t, p):
        pass


    def plot(self, X, Y, loc, var='0', icpt='icpt', ax=None):
        if ax is None:
            fig = plt.figure()
        else:
            plt.sca(ax)
        x, y, w = xr.broadcast(X.sel(var=0), Y, self.W.sel(space=loc[1]))
        plt.scatter(x, y, c=w)
        plt.colorbar()
        plt.scatter(x.sel(space=loc[1]), y.sel(space=loc[1]), color='r')
        xl = plt.gca().get_xlim()
        z = np.linspace(xl[0], xl[1], 100)
        a = self.p.sel(time=loc[0], space=loc[1], var=icpt).values
        b = self.p.sel(time=loc[0], space=loc[1], var=var).values
        plt.plot(z, a + b * z, '-')
        if ax is None:
            fig.show()

    @classmethod
    def test(cls, icpt=[[0, 1], [2, 3]], slope=[[3, -1], [1, -5]], dist=0, lapl=0, time=0, lda=0):
        x = np.atleast_3d(np.linspace(0, 1, 100)).transpose((1,0,2))
        y = [slope] * x + [icpt]
        y = y + np.random.randn(*y.shape) / 10
        x = (x * np.ones(np.shape(icpt))).transpose((1,0,2)).reshape((-1,2))
        # x = np.r_['0,3', x, np.ones(x.shape)]
        x = np.r_['0,3', x]
        X = xr.DataArray(x, dims=('var', 'time', 'space'))
        Y = xr.DataArray(y.transpose((1,0,2)).reshape((-1,2)), dims=('time', 'space'))
        I = np.identity(2)
        r = cls(1-I) #, lambda d:d+(1-I)*(1-dist), lambda d:I+(1-I)*(1-lapl))
        r.time_laplacian(size=2, func=lambda d:I+(1-I)*(1-time))
        r.regress(X, Y, lda, window=100, step=100)
        fig, axs = plt.subplots(r.p['time'].size, r.p['space'].size)
        for i,ay in enumerate(axs):
            for j, ax in enumerate(ay):
                # r.plot(X, Y, loc=(i,j), ax=ax, var=0, icpt=1)
                r.plot(X, Y, loc=(i,j), ax=ax)
        return r



if __name__=="__main__":
    D = pd.HDFStore('../../data/tables/station_data_new.h5')
    S = pd.HDFStore('../../data/tables/LinearLinear.h5')
    T = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, 'elev')) + 273.15
    Tm = S['T2n']
    sta = D['sta']
    dist = D['dist']
    dz = S['z']['d03_op'] - sta['elev']
    dt = Tm['d03_0_00']-T
    D.close()
    S.close()
    Y = xr.DataArray(dt, dims=('time', 'space')).dropna('time', 'all').dropna('space', 'all')
    x = dz.to_frame()
    x[1] = 1
    X = xr.DataArray(x, dims=('space', 'var')).dropna('space')
