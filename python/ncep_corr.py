#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from mpl_toolkits import basemap
import helpers as hh
import statsmodels.api as sm


def spline(x):
    m = x.groupby((x.index.year, x.index.month)).mean().dropna()
    j = pd.DatetimeIndex(['{}-{}'.format(*t) for t in m.index])
    i = np.array(j[1:-1] + np.diff(j[1:]) / 2, dtype='datetime64[h]').astype(float)
    cs = CubicSpline(i, m[1:-1], bc_type='natural')
    y = x[j[1]:j[-1]]
    t = np.array(y.index + pd.Timedelta('12H'), dtype='datetime64[h]').astype(float)
    return pd.DataFrame(y.as_matrix() - cs(t), index=y.index)


class RegressionMap(object):
    def __init__(self, nc, var='slp', timedelta='-4H'):
        self.map = basemap.Basemap(
            projection = 'merc',
            llcrnrlon = -180,
            llcrnrlat = -70,
            urcrnrlon = -60,
            urcrnrlat = 10
        )
        lon, lat = hh.lonlat(nc)
        self._i, self._j = self.map(lon-360, lat)
        slp = nc.variables[var][:]
        s = pd.DataFrame(slp.reshape((slp.shape[0], -1)))
        t = pd.DatetimeIndex(hh.get_time(nc) + pd.Timedelta(timedelta))
        self.slp = s.groupby(t.date).mean()
        self.t = pd.DatetimeIndex(self.slp.index)

    def regression(self, wind, lag=0, dir=0, pval=0, plot=True):
        """
        If dir=0, wind is predictor for slp field;
        if dir=0, slp is predictor (each point individually).
        """
        self.slp.index = self.t + pd.Timedelta(lag, 'D')
        c = pd.concat((wind, self.slp), 1, copy=True).dropna(0)
        # this just orders the columns for the following regression, so we know indeces from the matrix to take
        c.columns = np.arange(c.shape[1]) + 1
        c[0] = 1 # for regression intercept
        c.sort_index(1, inplace=True)
        X = c.as_matrix()
        if dir:
            if pval:
                r, p = zip(*[self.ols(X[:,[0,i]], X[:,1]) for i in range(2, X.shape[1])])
            else:
                r = [np.linalg.lstsq(X[:,[0,i]], X[:,1])[0][1] for i in range(2, X.shape[1])]
        else:
            if pval:
                r, p = zip(*[self.ols(X[:,:2], X[:,i]) for i in range(2, X.shape[1])])
            else:
                r = np.linalg.lstsq(X[:,:2],X[:,2:])[0][1]
        r = np.array(r).reshape(self._i.shape)
        if pval:
            p = np.array(p).reshape(self._i.shape)
            if plot:
                self.contourf(r, lag, p, pval)
            return r, p
        else:
            if plot:
                self.contourf(r, lag)
            return r

    def ols(self, X, Y):
        res = sm.OLS(Y,X).fit()
        return res.params[1], res.pvalues[1]

    def contourf(self, r, lag, p=None, pval=0):
        fig = plt.figure()
        plt.set_cmap('viridis')
        self.map.contourf(self._i, self._j, r)
        plt.colorbar()
        if pval:
            self.map.contour(self._i, self._j, p, [pval], colors=['w'], linewidths=[2])
        plt.gca().set_title('lag -{} days'.format(lag))
        fig.show()

    def lag_regressions(self, wind, lags, dir=0, pval=0, plot=True):
        if pval:
            self.regs, self.pvals = zip(*[self.regression(wind, l, dir, pval, plot) for l in lags])
        else:
            self.regs = [self.regression(wind, l, dir, plot=plot) for l in lags]
        self.lags = lags

    def plot_map(self, lvl, pval=0, st=None):
        # sm = cm.ScalarMappable(norm=colors.Normalize(vmin=min(z), vmax=max(z)))
        # sm.set_array(z)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color cycle
        fig = plt.figure()
        plt.set_cmap('Greys')
        self.map.drawcoastlines()
        self.map.contourf(self._i, self._j, self.regs[0], alpha=.5)
        # lvl = 150
        # lvl = .00005
        for i, r in enumerate(self.regs):
            c = self.map.contour(self._i, self._j, r, [lvl], colors=[colors[i]], linewidths=[2], linestyles=['solid'])
            plt.clabel(c, [lvl], fmt={lvl: '-{} days'.format(self.lags[i])})
            if pval:
                p = self.map.contour(self._i, self._j, self.pvals[i], [pval], colors=[colors[i]], linestyles=['dotted'])
        if st is not None:
            self.map.plot(st.lon, st.lat, 'ro', latlon=True)
        self.map.drawmeridians(range(-180,-50,20), labels=[0,0,0,1], color='w')
        self.map.drawparallels(range(-80,10,20), labels=[1,0,0,0], color='w')
        fig.show()
        return fig

class PCA(object):
    def __init__(self, data, stations):
        d = data.groupby(data.index.date).mean()
        d.index = pd.DatetimeIndex(d.index)
        self.sta = stations.loc[d.columns]
        self.Y = pd.concat([spline(x) for i,x in d.iteritems()], 1)
        self.Y.columns = d.columns
        self.Y -= self.Y.mean()
        self.mask = d.copy()
        self.mask[:] = 0
        self.mask[d.isnull()] = 1
        self.map = basemap.Basemap(
            lat_0=-30.5,
            lat_1=-10.0,
            lat_2=-40.0,
            lon_0=-71.0,
            projection='lcc',
            width=300000,
            height=400000,
            resolution='h'
        )

    def pca(self, rec=0, nrec=None, scale=False, covscale=None):
        y = self.Y.fillna(0)
        if rec:
            y += self.mask * self.R
        if nrec is None:
            nrec = self.Y.shape[1]
        if scale:
            self.cov = self.S.cov()
        else:
            self.cov = self.Y.cov()
        if covscale is not None:
            self.cov *= covscale
        w, v = np.linalg.eig(self.cov)
        i = np.argsort(w)[::-1] # eigenvalues not guaranteed to be ordered
        self.values = w[i]
        self.vectors = v[:,i]
        self.PC = y.dot(self.vectors[:,:nrec])
        self.R = self.PC.dot(self.vectors[:,:nrec].T)
        self.R.columns = self.Y.columns
        mse = ((self.R - self.Y)**2).sum().sum()
        print('mse: {}, explained: {}%'.format(mse, self.values[0] / np.sum(np.abs(w)) * 100))
        # r = pd.DataFrame(np.r_['1,2',t].T*u, index=t.index, columns=m.columns)

    def explained(self, n=None):
        n = self.values.shape[0] if n is None else n
        return self.values[:n] / np.sum(np.abs(self.values))

    def plot_map(self, pc, scale=1000):
        fig = plt.figure()
        self.map.scatter(*self.sta[['lon','lat']].as_matrix().T,
                         c = self.vectors[:,pc],
                         s = np.abs(self.vectors[:,pc]) * scale,
                         latlon=True)
        self.map.drawcoastlines()
        plt.colorbar()
        fig.show()

def plot(R, P, c1=None, c2=None, s=500):
    fig, ax = plt.subplots(2, 5, figsize=(15,6), subplotpars=SubplotParams(right=.95, left=.05))
    plt.set_cmap('coolwarm')
    for j in range(2):
        R.lag_regressions(P.PC[j], range(0, 8, 2), pval=.05, plot=False)
        for i in range(4):
            plt.sca(ax[j,3-i])
            if c1 is None:
                pl = R.map.contourf(R._i, R._j, R.regs[i])
                print(pl.get_clim())
            else:
                R.map.contourf(R._i, R._j, R.regs[i], c1)
            R.map.contour(R._i, R._j, R.pvals[i], [0.05], linewidths=[2], colors=['w'])
            R.map.plot(P.map.boundarylons, P.map.boundarylats, latlon=True, color='k')
            R.map.drawcoastlines()
            if j==0:
                plt.title('lag {} days'.format(-R.lags[i]))
        # bb = ax[0,3].get_position()
        # plt.colorbar(cax=fig.add_axes([bb.x1+0.02,bb.y0,0.05,bb.y1-bb.y0]))
        plt.sca(ax[j,4])
        pl = P.map.scatter(*P.sta[['lon','lat']].as_matrix().T,
                      c = P.vectors[:,j],
                      s = np.abs(P.vectors[:,j]) * s,
                      latlon=True)
        P.map.drawcoastlines()
        print(pl.get_clim())
        if c2 is not None:
            plt.clim(-c2, c2)
    fig.show()
    return fig, ax


if __name__ == "__main__":
    from netCDF4 import Dataset

    # nc = Dataset('../../data/NCEP/X164.77.119.34.78.12.40.44.nc')
    # R = RegressionMap(nc)

    # D = pd.HDFStore('../../data/tables/station_data_new.h5')
    # vv = D['vv_ms'].xs('prom',1,'aggr')

    def one():
        st = D['sta'].loc['4']
        v4 = vv['4']
        vd = dti(v4.groupby(v4.index.date).mean())
        vd.columns = v4.columns.get_level_values('elev')
        y = spline(vd['5']['2012-07-23':])
        regs = Rmap.lag_regressions(y, range(0, 8, 2), 1)
        Rmap.plot_map(regs, 5e-5)

    def pc():
        # remove multiple sensors at same station
        vv.drop(['PAZVV5', '117733106', 'RMRVV5', 'VCNVV5', 'PEVV5', 'RMPVV5', 'CGRVV10',
                 'CHPLVV10', 'LCARVV5', 'LLHVV10M', 'QSVV1', 'VLLVV30'], 1, 'code', inplace=True)
        vv.columns = vv.columns.get_level_values('station')

        # remove suspicious data
        vv.drop('9', 1, inplace=True)
        vv['INIA66'][:'2012-03-20'] = np.nan

        # remove far away data
        vv.drop(['CNPW', 'CNSD'], 1, inplace=True)

        d = vv['2013':]
        d = d.loc[:, d.count() > .7 * len(d)]

    dist = D['dist'][d.columns].loc[d.columns]

    P = PCA(d, D['sta'])

    # P.pca(scale=dist)
    # P.pca(1, scale=dist)
    P.pca()
    P.pca(1)

