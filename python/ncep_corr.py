#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import cm, colors


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

    def reg(self, wind, lag=0, dir=0):
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
            r = [np.linalg.lstsq(X[:,[0,i]], X[:,1])[0][1] for i in range(2, X.shape[1])]
            return lag, np.array(r).reshape(self._i.shape)
        else:
            return lag, np.linalg.lstsq(X[:,:2],X[:,2:])[0][1].reshape(self._i.shape)

    def lag_regressions(self, wind, lags, dir=0):
        return [self.reg(wind, l, dir) for l in lags]

    def plot_map(self, regressions, lvl):
        # sm = cm.ScalarMappable(norm=colors.Normalize(vmin=min(z), vmax=max(z)))
        # sm.set_array(z)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default color cycle
        fig = plt.figure()
        plt.set_cmap('Greys')
        self.map.drawcoastlines()
        self.map.contourf(self._i, self._j, regressions[0][1], alpha=.5)
        # lvl = 150
        # lvl = .00005
        for i, r in enumerate(regressions):
            c = self.map.contour(self._i, self._j, r[1], [lvl], colors=[colors[i]])
            plt.clabel(c, [lvl], fmt={lvl: '-{} days'.format(r[0])})
        self.map.plot(st.lon, st.lat, 'ro', latlon=True)
        self.map.drawmeridians(range(-180,-50,20), labels=[0,0,0,1], color='w')
        self.map.drawparallels(range(-80,10,20), labels=[1,0,0,0], color='w')
        fig.show()
        return fig


def spline(x):
    m = x.groupby((x.index.year, x.index.month)).mean().dropna()
    j = pd.DatetimeIndex(['{}-{}'.format(*t) for t in m.index])
    i = np.array(j[1:-1] + np.diff(j[1:]) / 2, dtype='datetime64[h]').astype(float)
    cs = CubicSpline(i, m[1:-1], bc_type='natural')
    y = x[j[1]:j[-1]]
    t = np.array(y.index + pd.Timedelta('12H'), dtype='datetime64[h]').astype(float)
    return pd.DataFrame(y.as_matrix() - cs(t), index=y.index)


def pca(d, r=None, n=d.shape[1], npc=1, scale=False):
    m = d.copy()
    m[:] = 0
    m[d.isnull()] = 1
    if r is None:
        y = d.fillna(0)
    else:
        y = d.fillna(0) + m * r
    c = d.cov()
    if scale:
        c *= dist
    w, v = np.linalg.eig(c)
    i = np.argsort(w) # eigenvalues not guaranteed to be ordered
    t = y.dot(v[:,i[:n]])
    r = t.dot(v[:,i[:n]].T)
    r.columns = d.columns
    mse = ((r-d)**2).sum().sum()
    print(mse)
    # r = pd.DataFrame(np.r_['1,2',t].T*u, index=t.index, columns=m.columns)
    return t.iloc[:,i[-npc:]], r


if __name__ == "__main__":
    from netCDF4 import Dataset
    import helpers as hh
    import matplotlib.pyplot as plt
    from mpl_toolkits import basemap

    # nc = Dataset('../../data/NCEP/X164.77.119.34.78.12.40.44.nc')
    Rmap = RegressionMap(nc)

    # D = pd.HDFStore('../../data/tables/station_data_new.h5')
    vv = D['vv_ms'].xs('prom',1,'aggr')

    def one():
        st = D['sta'].loc['4']
        v4 = vv['4']
        vd = dti(v4.groupby(v4.index.date).mean())
        vd.columns = v4.columns.get_level_values('elev')
        y = spline(vd['5']['2012-07-23':])
        regs = Rmap.lag_regressions(y, range(0, 8, 2), 1)
        Rmap.plot_map(regs, 5e-5)

    # remove multiple sensors at same station
    vv.drop(['PAZVV5', '117733106', 'RMRVV5', 'VCNVV5', 'PEVV5', 'RMPVV5', 'CGRVV10',
             'CHPLVV10', 'LCARVV5', 'LLHVV10M', 'QSVV1', 'VLLVV30'], 1, 'code', inplace=True)
    vv.columns = vv.columns.get_level_values('station')

    # remove suspicious data
    vv.drop('9', 1, inplace=True)
    vv['INIA66'][:'2012-03-20'] = np.nan

    d = vv['2013':]
    d = d.loc[:, d.count() > .7 * len(d)]
    dm = dti(d.groupby(d.index.date).mean())

    dist = D['dist'][d.columns].loc[d.columns]
    sta = D['sta'].loc[d.columns]

    s = pd.concat([spline(x) for i,x in dm.iteritems()], 1)
    s.columns = dm.columns
    s -= s.mean()

    t, r = pca(s)
    t, r = pca(s, r)

    regs = Rmap.lag_regressions(t, range(0, 8, 2))


