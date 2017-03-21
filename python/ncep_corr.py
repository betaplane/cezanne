#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import cm, colors

def dti(df):
    try:
        df.index = pd.DatetimeIndex(df.index)
    except:
        df.index = pd.DatetimeIndex(['{}-{}'.format(*a) for a in df.index])
    return df

def reg(wind, slp, lag=0):
    sc = slp.copy()
    sc.index += pd.Timedelta(lag, 'D')
    c = pd.concat((wind, sc), 1, copy=True).dropna(0)
    c.columns = np.arange(c.shape[1]) + 1
    c[0] = 1 # for regression intercept
    c.sort_index(1, inplace=True)
    X = c.as_matrix()
    # return np.linalg.lstsq(X[:,:2],X[:,2:])[0][1]
    return np.array([np.linalg.lstsq(X[:,[0,i]], X[:,1])[0][1] for i in range(2, X.shape[1])])


def tfloat(t, dt=0):
    return np.array(t + pd.Timedelta(dt), dtype='datetime64[h]').astype(float)

def spline(x):
    m = dti(x.groupby((x.index.year, x.index.month)).mean())
    i = tfloat(m.index[1:-1] + np.diff(m.index[1:]) / 2)
    cs = CubicSpline(i, m[1:-1], bc_type='natural')
    y = x[m.index[1]:m.index[-1]]
    return pd.DataFrame(y.as_matrix() - cs(tfloat(y.index, '12H')), index=y.index)


if __name__ == "__main__":
    from netCDF4 import Dataset
    import helpers as hh
    import matplotlib.pyplot as plt
    from mpl_toolkits import basemap

    nc = Dataset('../../data/NCEP/X164.77.119.34.78.12.40.44.nc')
    t = hh.get_time(nc)
    lon,lat = hh.lonlat(nc)
    slp = nc.variables['slp'][:]
    s = pd.DataFrame(slp.reshape((slp.shape[0], -1)), index=t)
    d = dti(s.groupby(s.index.date).mean())

    D = pd.HDFStore('../../data/tables/station_data_new.h5')
    st = D['sta'].loc['4']
    v = D['vv_ms'].xs('prom',1,'aggr')
    v4 = v['4']
    v4.index -= pd.Timedelta('4H')
    vd = dti(v4.groupby(v4.index.date).mean())
    vd.columns = v4.columns.get_level_values('elev')
    x5 = vd['5']['2012-07-23':]

    y = spline(x5)

    z = range(0,8,2)
    R = [reg(y, d, lag=i) for i in z]
    m = basemap.Basemap(
        projection = 'merc',
        llcrnrlon = -180,
        llcrnrlat = -70,
        urcrnrlon = -60,
        urcrnrlat = 10
    )

    # sm = cm.ScalarMappable(norm=colors.Normalize(vmin=min(z), vmax=max(z)))
    # sm.set_array(z)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure()
    plt.set_cmap('Greys')
    m.drawcoastlines()
    m.contourf(lon, lat, R[0].reshape(slp.shape[1:]), latlon=True, alpha=.5)
    # lvl = 150
    lvl = .00005
    for i, r in enumerate(R):
        c = m.contour(lon, lat, r.reshape(slp.shape[1:]), [lvl], colors=[colors[i]], latlon=True)
        plt.clabel(c, [lvl], fmt={lvl: '-{} days'.format(z[i])})
    fig.show()
    m.plot(st.lon, st.lat, 'ro', latlon=True)
    m.drawmeridians(range(-180,-50,20), labels=[0,0,0,1], color='w')
    m.drawparallels(range(-80,10,20), labels=[1,0,0,0], color='w')
