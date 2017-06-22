#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import SubplotParams
from mpl_toolkits import basemap
import statsmodels.api as sm
import xarray as xr


def spline(x):
    m = x.groupby((x.index.year, x.index.month)).mean().dropna()
    j = pd.DatetimeIndex(['{}-{}'.format(*t) for t in m.index])
    i = np.array(j[1:-1] + np.diff(j[1:]) / 2, dtype='datetime64[h]').astype(float)
    cs = CubicSpline(i, m[1:-1], bc_type='natural')
    y = x[j[1]:j[-1]]
    t = np.array(y.index + pd.Timedelta('12H'), dtype='datetime64[h]').astype(float)
    return pd.DataFrame(y.as_matrix() - cs(t), index=y.index)


class RegressionMap(object):
    def __init__(self, nc_file, timedelta=None):
        self.map = basemap.Basemap(
            projection = 'merc',
            llcrnrlon = -180,
            llcrnrlat = -70,
            urcrnrlon = -60,
            urcrnrlat = 10
        )
        with xr.open_dataset(nc_file) as nc:
            self.Y = nc[var].load()
        if timedelta is not None:
            self.Y['time'] = self.Y.time + pd.Timedelta(timedelta)
        if ave is not None:
            self.Y = self.average(self.Y, ave)

    @staticmethod
    def average(x, period):
        # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#anchored-offsets
        if period == 'year':
            return x.resample('AS', dim='time', how='mean') # 'annual start'
        if period == 'month':
            return x.resample('MS', dim='time', how='mean') # 'monthly start'
        if period == 'season':
            return x.resample('QS-Mar', dim='time', how='mean') # 'qarterly start March'


    def regression(self, series, predictor='series', intercept=True, lag=None, pval=False):
        """
        If predictor='series', time series is predictor for space-time field, otherwise the other way around.
        """
        if lag is not None:
            Y['time'] = Y.time + pd.Timedelta(lag)
        Y, x = xr.align(self.Y.stack(space = ('lon', 'lat')).squeeze(),
                        xr.DataArray(series).rename({'dim_0': 'time'}))
        X = sm.tools.add_constant(x) if intercept else np.array(x, ndmin=2).T

        if dir == 'series':
            if pval:
                def ols(Y, X):
                    res = sm.OLS(Y, X).fit()
                    return res.params[1], res.pvalues[1]
                r, p = np.apply_along_axis(ols, 0, np.array(Y), X=X)
                self.p = xr.DataArray(p, coords=[Y.space]).unstack('space')
            else:
                r = np.linalg.lstsq(X, Y)[0][1]
        else:
            if pval:
                def ols(X, Y):
                    if intercept:
                        X = sm.tools.add_constant(X)
                    res = sm.OLS(Y, X).fit()
                    return res.params[1], res.pvalues[1]
                r, p = np.apply_along_axis(ols, 0, np.array(Y), Y=X[:, -1:])
                self.p = xr.DataArray(p, coords=[Y.space]).unstack('space')
            else:
                def lstsq(y, x):
                    return np.linalg.lstsq(x, y)[0][1]
                r = np.apply_along_axis(lstsq, 0, Y, x=X)
        self.r = xr.DataArray(r, coords=[Y.space]).unstack('space')

    def contourf(self, pval=0, lvls=None):
        lon, lat = np.meshgrid(self.r.lon - 360, self.r.lat)
        fig = plt.figure()
        plt.set_cmap('coolwarm')
        if lvls is None:
            self.map.contourf(lon, lat, self.r.values.T, latlon=True)
        else:
            self.map.contourf(lon, lat, self.r.values.T, lvls, latlon=True)
        self.map.drawcoastlines()
        plt.colorbar()
        if pval:
            self.map.contour(lon, lat, self.p.values.T, [pval], colors=['w'], linewidths=[2], latlon=True)
        fig.show()

if __name__ == "__main__":
    pass
