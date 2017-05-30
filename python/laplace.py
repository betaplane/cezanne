#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg as la
import helpers as hh
import matplotlib.pyplot as plt


class reg_base(object):
    def __init__(self, dz, dist):
        self._dz = dz.dropna()
        i = self._dz.index
        self._dist = dist.loc[i, i]



# This is an implementation of a linear regression local in space and time (but over some neighborhood), with
# spatial smoothness of the regression coefficients enforced by regularization via a spatial graph Laplacian.
# Based on:
# Subbian, Karthik, and Arindam Banerjee. “Climate Multi-Model Regression Using Spatial Smoothing.”
# In SDM, 324–332. SIAM, 2013. http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.36.

class GLR(object):
    def __init__(self, reg_mask, coeff_mask=None):
        self._w = xr.DataArray(reg_mask, dims=('space', 'i'))
        self.W = 1 - self._w / self._w.max()
        self.L = np.diag(self.W.sum('i')) - self.W

    def regress(self, X, Y, lda, plot=None):
        """
        :params X: predictor (var x space x time)
        :params Y: target (var x space x time)
        :params lda: regularization parameter (lambda)
        """
        x, y = xr.broadcast(X * self.W, Y * self.W, exclude=['var'])
        C = (x * y).sum(('time', 'space')).transpose('i', 'var')
        a = x.transpose('i','var','space','time')
        A = la.block_diag(*np.einsum('ijkl,imkl->ijm', a, a))
        i = a['i']
        L = np.kron(self.L.loc[i,i], np.identity(a['var'].size))
        p = la.solve(A + lda * L, np.array(C).flatten()).reshape(a.shape[:2])
        self.p = xr.DataArray(p, coords=[a['i'], a['var']], dims=['i', 'var'])
        if plot is not None:
            self.plot(X, Y, plot)

    def plot(self, X, Y, loc):
        fig = plt.figure()
        x, y, w = xr.broadcast(X.sel(var=0), Y, self.W.sel(i=loc))
        plt.scatter(x, y, c=w)
        plt.colorbar()
        plt.scatter(x.sel(space=loc), y.sel(space=loc), color='r')
        xl = plt.gca().get_xlim()
        z = np.linspace(xl[0], xl[1], 100)
        plt.plot(z, self.p.sel(i=loc, var=0).values + z * self.p.sel(i=loc, var=0).values, '-')
        fig.show()




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
