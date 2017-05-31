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
    def __init__(self, reg_mask, coeff_mask=None):
        self.W = xr.DataArray(reg_mask, dims=('space', 'i'))
        # self.W = 1 - self._w / self._w.max()
        self.L = xr.DataArray(np.diag(np.array(coeff_mask.sum(1))) - coeff_mask, dims=('space', 'i'))

    def regress(self, X, Y, lda, plot=None):
        """
        :param X: predictor
        :type X: xarray with dims 'var', 'space', 'time'
        :param Y: target
        :type Y: xarray with dims 'space', 'time'
        :param lda: regularization parameter (lambda)
        :param plot: None or location from 'space' dim to plot
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
        plt.plot(z, self.p.sel(i=loc, var=1).values + z * self.p.sel(i=loc, var=0).values, '-')
        fig.show()

    @classmethod
    def test(cls, icpt=[0, 1], slope=[3, 3], reg=0, coeff=0, lda=0, plot=0):
        x = np.linspace(0, 1, len(icpt)*100).reshape((100, -1))
        y = [slope] * x + np.random.rand(len(icpt)*100).reshape(x.shape) / 2 + [icpt]
        x = np.r_['0,3', x, np.ones(x.shape)]
        X = xr.DataArray(x, dims=('var', 'time', 'space'))
        Y = xr.DataArray(y, dims=('time', 'space'))
        I = np.identity(len(icpt))
        r = cls(I+(1-I)*(1-reg), I+(1-I)*(1-coeff))
        r.regress(X, Y, lda, plot)
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
