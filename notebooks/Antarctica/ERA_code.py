import xarray as xr
import numpy as np

earth_radius = 6367470

def area_mean(x):
    dy = earth_radius * np.pi / 180 * 0.75
    dx = np.cos(x.lat * np.pi / 180) * earth_radius * np.pi / 180 * 0.75
    nx = len(x.lon)
    area = dx * xr.DataArray(np.repeat(dy, nx), coords=[('lon', x.lon)])
    return (x * area).sum(['lon', 'lat']).squeeze() / np.sum(area)

def box_fluxes(u, v):
    dy = earth_radius * np.pi / 180 * 0.75
    dx = np.cos(u.lat * np.pi / 180) * earth_radius * np.pi / 180 * 0.75
    du = u.isel(lon=[0, -1]).diff('lon') * dy
    dv = (v * dx).isel(lat=[-1, 0]).diff('lat')

    area = dx * xr.DataArray(np.repeat(dy, len(u.lon)), coords=[('lon', dv.lon)])
    sum_a = np.sum(area)
    du = du.sum('lat').squeeze() / sum_a
    dv = dv.sum('lon').squeeze() / sum_a
    return du, dv

def section_fluxes(u, v):
    dy = earth_radius * np.pi / 180 * 0.75
    dx = np.cos(u.lat * np.pi / 180) * earth_radius * np.pi / 180 * 0.75

    du = u.isel(lon=[0, -1]).diff('lon') / dx / (len(u.lon) - 1)
    dv = - v.diff('lat', 1).mean('lon') / dy
    dv['lat'] = dv.lat + .75/2
    return du.squeeze(), dv.squeeze()

def monthly(x):
    import pandas as pd
    t = x.indexes['time']
    x.coords['month'] = ('time', pd.DatetimeIndex('{}-{}'.format(*x) for x in zip(t.year, t.month)))
    return x.groupby('month')

def trend(x, time='time', detrend=False):
    """returns trend per year"""
    from statsmodels.tools import add_constant
    year =  3600 * 24 * 365.24 # slope is w.r.t. seconds
    t = add_constant(x[time].values.astype('datetime64[s]').astype(float))
    lsq = np.linalg.lstsq(t, x.squeeze())[0]
    coords = [c for c in set(x.coords) - {time} if (x.coords[c].shape != ()) and (len(x.coords[c]) > 1)]
    if len(coords) == 1:
        if detrend:
            return xr.DataArray(x - t.dot(lsq), coords=[x.time, x.coords[coords[0]]])
        return xr.DataArray(lsq[1, :], coords=[x.coords[coords[0]]]) * year
    elif len(coords) == 0:
        return x - t.dot(lsq) if detrend else lsq[1] * year
    else:
        raise Exception('more than one additional coordinate')

def cube(x):
    import iris
    lat = iris.coords.DimCoord(x.lat, standard_name='latitude', units='degrees')
    lon = iris.coords.DimCoord(x.lon, standard_name='longitude', units='degrees')
    return iris.cube.Cube(x.squeeze(), dim_coords_and_dims=[(lat, 0), (lon, 1)])

def flux_decomp(u, v, q, box, time_grouper):
    """q should be pressure-weighted"""

    # "bar/prime"
    bp = lambda g: (g.mean('time'), g.apply(lambda x: x - x.mean('time')))

    u_bar, u_prime = bp(time_grouper(u.sel(**box)))
    v_bar, v_prime = bp(time_grouper(v.sel(**box)))
    q_bar, q_prime = bp(time_grouper(q.sel(**box)))

    du_bar, dv_bar = [x.sum('lev') / 9.8 for x in area_flxs(u_bar * q_bar, v_bar * q_bar)]
    du_prime, dv_prime = [x.sum('lev') / 9.8 for x in area_flxs(u_prime * q_prime, v_prime * q_prime)]
    return du_bar, du_prime, dv_bar, dv_prime
