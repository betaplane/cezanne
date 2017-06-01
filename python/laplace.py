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
    def __init__(self, reg_mask, coeff_mask=None, time_mask=None):
        self.W = xr.DataArray(reg_mask, dims=('space', 'i'))
        # self.W = 1 - self._w / self._w.max()
        L = np.kron(time_mask, coeff_mask)
        i = pd.MultiIndex.from_product(([0,1], [0,1]))
        self.L = xr.DataArray(np.diag(np.array(L.sum(1))) - L, coords=[i, i])

    def _block(self, X, Y):
        x, y = xr.broadcast(X * self.W, Y * self.W, exclude=['var'])

        # instead of fitting intercept (not desirable for regularized regression)
        # remove mean from predictors, estimate intercept as mean of target
        # (see Hastie et al.)
        xm = (x.sum('space') / self.W.sum('space')).mean('time')
        ym = (y.sum('space') / self.W.sum('space')).mean('time')
        # from IPython.core.debugger import Tracer; Tracer()()
        x = x - xm * self.W # important!
        y = y - ym * self.W # mean needs to be multiplied by mask
        C = (x * y).sum(('time', 'space')).transpose('i', 'var')
        a = x.transpose('i','var','space','time')
        A = la.block_diag(*np.einsum('ijkl,imkl->ijm', a, a))
        return A, C, xm, ym

    def regress(self, X, Y, lda, window=None, step=None):
        """
        :param X: predictor
        :type X: xarray with dims 'var', 'space', 'time' (NO intercept!)
        :param Y: target
        :type Y: xarray with dims 'space', 'time'
        :param lda: regularization parameter (lambda)
        """
        i = pd.MultiIndex.from_product(([0, 1], X['space']))
        nvar = X['var'].size
        L = self.L.loc[i, i]
        # from IPython.core.debugger import Tracer; Tracer()()
        A, C, xm, ym = zip(*[self._block(
            X.isel(time=slice(k, k+window)),
            Y.isel(time=slice(k, k+window))
        ) for k in range(0, Y['time'].size, step)])
        L = np.kron(L, np.identity(nvar))
        A = la.block_diag(*A)
        C = np.vstack(C).flatten()
        p = la.solve(A + lda * L, C).reshape((-1, nvar)) # i x nvar
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
    def test(cls, icpt=[[0, 1], [2, 3]], slope=[[3, -1], [1, -5]], reg=0, coeff=0, time=0, lda=0):
        x = np.atleast_3d(np.linspace(0, 1, 100)).transpose((1,0,2))
        y = [slope] * x + [icpt]
        y = y + np.random.randn(*y.shape) / 10
        x = (x * np.ones(np.shape(icpt))).transpose((1,0,2)).reshape((-1,2))
        # x = np.r_['0,3', x, np.ones(x.shape)]
        x = np.r_['0,3', x]
        X = xr.DataArray(x, dims=('var', 'time', 'space'))
        Y = xr.DataArray(y.transpose((1,0,2)).reshape((-1,2)), dims=('time', 'space'))
        I = np.identity(2)
        r = cls(I+(1-I)*(1-reg), I+(1-I)*(1-coeff), I+(1-I)*(1-time))
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
