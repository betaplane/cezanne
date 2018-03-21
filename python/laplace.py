"""
Note
----

The 'file' field in the bibtex file caused problems for sphinxcontrib.bibtex.

"""
#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import scipy.linalg as la
import matplotlib.pyplot as plt


class LRLR(object):
    """Laplacian-regularized local regression: combines space-time localized regression (i.e., weights over a spatial and temporal neighborhood can be specified) with graph-Laplacian regularization of the regression coefficients (also in space and time). If no callables are given for weights and laplacian, both will be set to the linear inverse of the distances normalized to 1. See :cite:`subbian_climate_2013` for details on the graph-Laplacian regularization.

    Usage example::

        DL = data.CEAZA.Downloader()
        stations = DL.get_stations(fields=False)
        LR = LRLR(stations)

    :param stations: 'stations' DataFrame as returned from :meth:`.data.CEAZA.Downloader.get_stations`
    :type stations: :class:`~pandas.DataFrame`
    :param weights: function to apply to distances for spatial weighting of regression
    :type weights: callable or None
    :param laplacian: function to apply to distances for weighting of the spatial Laplacian

    .. bibliography:: refs.bib

    """
    def __init__(self, stations, weights=None, laplacian=None):
        d = self.distance_matrix(stations)
        if weights is None:
            self.D = 1 - d / d.max()
        else:
            self.D = weights(d)
        if laplacian is None:
            self.L = self.D
        else:
            self.L = laplacian(d)

    @staticmethod
    def distance_matrix(stations):
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        d = pd.DataFrame()
        for i, a in stations.iterrows():
            for j, b in stations.iterrows():
                d.loc[i, j] = g.inv(float(a.lon), float(a.lat), float(b.lon), float(b.lat))[2]
        return xr.DataArray(d, dims=('space', 'i'))


    def _block(self, X, Y, t):
        # nuke time indexes here so that broadcasted multiplication with weights works
        # for any time window (otherwise, would only multiply with same indexes)
        T = min(t, Y['time'].size)

        try:
            X['time'] = np.arange(T) # will fail if 'time' not in dims
        except ValueError:
            pass
        Y['time'] = np.arange(T)

        # get a missing values mask for the weight matrix
        mv = (X.count('var') == X['var'].size) * Y.notnull()

        # safeguard against window over-running end of time series, and zero out missing values
        W = self.W.sel(time=slice(0, T)) * mv

        # takes care of broadcasting too
        x = X * W
        y = Y * W
        # instead of fitting intercept (not desirable for regularized regression)
        # remove mean from predictors, estimate intercept as mean of target
        # (see Hastie et al.)
        w = W.sum(('space', 'time'))
        xm = x.sum(('space', 'time')) / w
        ym = y.sum(('space', 'time')) / w

        x = x - xm * W # important!
        y = y - ym * W # mean needs to be multiplied by mask
        C = (x * y).sum(('time', 'space')).transpose('i', 'var') # NOTE: 'i' dim comes from W / self.W
        # from IPython.core.debugger import Tracer; Tracer()()
        a = x.transpose('i', 'var', 'space', 'time')
        A = la.block_diag(*np.einsum('ijkl,imkl->ijm', a, a))
        return A, C, xm.transpose('i', 'var'), ym

    def regress(self, X, Y, lda, window=None, step=None, laplacian=None):
        """
        :param X: predictor
        :type X: xarray with dims 'time', 'space', 'var' (NO intercept!)
        :param Y: target
        :type Y: xarray with dims 'time', 'space'
        :param lda: regularization parameter (lambda)
        :param window: window width (in indexes) for weighting in time, or array of weights
        :type window: int or collection or None
        :param step: increment by which to shift the window (in indexes)
        :param laplacian: set if the time Laplacian should be computed from the data's time axis
        :type laplacian: callable
        """
        self._var = X['var']

        # use only spatial locations common to predictors and targets
        i = np.intersect1d(X.space, Y.space)
        i.sort()

        try:
            w = len(window)
        except TypeError:
            w = window if np.isscalar(window) else Y['time'].size
            self.W = self.D.loc[i, i] * xr.DataArray(np.ones(w), dims=('time'))
        else:
            self.W = self.D.loc[i, i] * xr.DataArray(window, dims=('time'))

        def sel(x, k, w):
            try:
                return x.isel(time=slice(k, k+w))
            except ValueError:
                return x

        # we prepare one time 'chunk' at a time - we need all the chunks eventually for
        # the linear system, but the subroutine only takes as much as needed as per time weighting
        # - but it does deal with all space at a time, both for weighting and arranging it for
        # the graph Laplacian
        r = range(0, Y['time'].size, Y['time'].size if step is None else step)
        A, C, xm, ym = zip(*[self._block(sel(X, k, w), sel(Y, k, w), w) for k in r])

        T = 1 - la.toeplitz(r) / len(r) # shold be 1 if len(r)==1
        if laplacian is not None:
            T = laplacian(T)
        L = np.kron(T, self.L.sel(space=i, i=i))
        self.LPL = np.kron(np.diag(L.sum(1)) - L, np.identity(self._var.size))

        self.A = la.block_diag(*A)
        self.C = np.vstack(C).flatten()
        self.ymean = np.vstack(ym).reshape((-1, 1))
        self.xmean = np.vstack(xm) # i x nvar
        self.compute(lda)

    def compute(self, lda):
        nvar = self._var.size
        i = self.W['space']
        p = la.solve(self.A + lda * self.LPL, self.C).reshape((-1, nvar)) # i x nvar

        # the intercept is computed from mean of y plus mean x dot beta
        icpt = self.ymean - np.sum(self.xmean * p, 1).reshape((-1,1))
        p = np.r_['1,2', p, icpt].reshape(( -1, i.size, nvar + 1)) # time x space x nvar + 1
        self.p = xr.DataArray(p, dims=['time', 'space', 'var'])
        self.p['space'] = i
        self.p['var'] = np.r_[self._var, ['icpt']]


    def plot(self, X, Y, loc, var='x', icpt='icpt', ax=None):
        if ax is None:
            fig = plt.figure()
        else:
            plt.sca(ax)
        # x, y, w = xr.broadcast(X.sel(var=var), Y, self.W.sel(space=loc[1]))
        # plt.scatter(x, y, c=w)
        x, y = xr.broadcast(X.sel(var=var), Y)
        plt.scatter(x, y)
        # plt.colorbar()
        plt.scatter(x.sel(space=loc[1]), y.sel(space=loc[1]), color='r')
        xl = plt.gca().get_xlim()
        z = np.linspace(xl[0], xl[1], 100)
        a = self.p.sel(time=loc[0], space=loc[1], var=icpt).values
        b = self.p.sel(time=loc[0], space=loc[1], var=var).values
        plt.plot(z, a + b * z, '-')
        if ax is None:
            fig.show()

    @classmethod
    def test(cls, icpt=[[0, 1], [2, 3]], slope=[[3, -1], [1, -5]], dist=0, lapl=0, time=0, lda=0, init=False, vars=False):
        x = np.atleast_3d(np.linspace(0, 1, 100)).transpose((1,0,2))
        y = [slope] * x + [icpt]
        y = y + np.random.randn(*y.shape) / 10
        x = (x * np.ones(np.shape(icpt))).transpose((1,0,2)).reshape((-1,2))
        # x = np.r_['0,3', x, np.ones(x.shape)]
        x = np.r_['0,3', x]
        X = xr.DataArray(x, dims=('var', 'time', 'space'))
        X['var'] = ['x']
        Y = xr.DataArray(y.transpose((1,0,2)).reshape((-1,2)), dims=('time', 'space'))
        Y[np.random.randint(0,200,50), :] = np.nan
        if vars:
            return X, Y
        I = np.identity(2)
        r = cls(1-I, lambda d:d*dist+I, lambda d:d*lapl+I)
        if init:
            return X, Y, r
        r.regress(X, Y, lda, window=100, step=100, laplacian=lambda d:I+(1-I)*time)
        fig, axs = plt.subplots(r.p['time'].size, r.p['space'].size)
        for i,ay in enumerate(axs):
            for j, ax in enumerate(ay):
                # r.plot(X, Y, loc=(i,j), ax=ax, var=0, icpt=1)
                r.plot(X, Y, loc=(i,j), ax=ax)
        return r

    @staticmethod
    def gauss(radius):
        "Radius is half width at half maximim (HWHM)"
        return lambda x: np.exp(-2 * np.log(2) * (x / radius)**2)

    def param(self, name):
        return self.p.sel(var=name).to_dataframe(name)[name].unstack()


if __name__=="__main__":
    import helpers as hh
    D = hh.data('station_data_new.h5')
    S = hh.data('LinearLinear.h5')
    t = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, level='elev')) + 273.15
    Tm = S['T2n']
    sta = D['sta']
    dist = D['dist']
    dz = S['z']['d03_op'] - sta['elev']
    dt = Tm['d03_0_00']-t
    D.close()
    S.close()
    T = xr.DataArray(t, dims=('time', 'space')).dropna('time', 'all').dropna('space', 'all')
    Z = xr.DataArray(sta['elev'].to_frame(), dims=('space', 'var')).dropna('space')

    Y = xr.DataArray(dt, dims=('time', 'space')).dropna('time', 'all').dropna('space', 'all')
    X = xr.DataArray(dz.to_frame(), dims=('space', 'var')).dropna('space')
