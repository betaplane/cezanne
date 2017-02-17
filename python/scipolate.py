#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate as ip
from mapping import affine


def grid_interp(xy, data, ij, index, time=None, method='nearest'):
    x, y = xy
    i, j = ij
    m, n = data.shape[-2:]
    tg = affine(x, y)
    coords = np.roll(tg(np.r_['0,2', [i, j]]).T, 1, 1)
    if time is None:
        return pd.Series(
            ip.interpn(
                (range(m), range(n)), data, coords, method,
                bounds_error=False),
            index=index)
    else:
        return pd.DataFrame(
         [ip.interpn((range(m),range(n)),data[k,:,:],coords,method,bounds_error=False) \
          for k in range(len(time))],
         index=time, columns=index
        )


def interp4D(xy, data, ij, index, t, method='nearest'):
    x, y = xy
    i, j = ij
    mn = (tuple(range(data.shape[-2])), tuple(range(data.shape[-1])))
    tg = affine(x, y)
    coords = np.roll(tg(np.r_['0,2', [i, j]]).T, 1, 1)
    a = dict([(l,dict([(t[k], ip.interpn(mn, data[k,l,:,:], coords, method, bounds_error=False)) \
     for k in range(data.shape[0])])) for l in range(data.shape[1])])
    p = pd.Panel(a).transpose(1, 2, 0)
    p.items = index
    return p


def irreg_interp(xy, data, ij, index, time=None, method='nearest'):
    x, y = xy
    i, j = ij
    interpolator = {
        'nearest': ip.NearestNDInterpolator,
        'linear': ip.LinearNDInterpolator
    }
    if time is None:
        itrp = interpolator[method]((x.flatten(), y.flatten()), data.flatten())
        return pd.Series(itrp(i, j), index=index)
    else:
        df = []
        for k in range(len(time)):
            print(k)
            itrp = interpolator[method](
                (x.flatten(), y.flatten()), data[k, :, :].flatten())
            df.append(itrp(i, j))
        return pd.DataFrame(df, index=time, columns=index)


def scipy_interp3(sta, map, lon, lat, data, time=None):
    m, n = data.shape
    x, y, i, j = llm(grid, sta)
    tg = affine(x, y)
    coords = tg(np.r_['0,2', [i, j]]).T
    j, i = np.mgrid[:m, :n]
    if time is None:
        interp = ip.LinearNDInterpolator(
            (i.flatten(), j.flatten()), data.flatten())
        return pd.Series(interp(coords), index=sta.index)
    else:
        pass


if __name__ == "__main__":
    from mapping import basemap, affine

    K = 273.15
    origin = np.datetime64('2005-01-01T00', 'h')

    store = pd.HDFStore('data/data.h5')
    sta = store['sta']

    g = Dataset('data/wrf/geo_em.d02.nc')
    d = Dataset('data/wrf/wrf_d02_t2_2014-09-10T00.nc')
    T = d.variables['T2'][:, :, :] - K
    time = d.variables['time'][:].astype('timedelta64[h]') + origin
    lon = g.variables['XLONG_M'][0, :, :]
    lat = g.variables['XLAT_M'][0, :, :]

    m = basemap(g)
    # 	dint = pd.concat((
    # 		scipy_interp1(sta, m, lon, lat, T2),
    # 		scipy_interp2(sta, m, lon, lat, T2),
    # 		scipy_interp3(sta, m, lon, lat, T2)
    # 	), axis=1)

    # 	T1 = scipy_interp1(sta, g, T, time=time)
    T2 = scipy_interp2(sta, m, lon, lat, T, time)
