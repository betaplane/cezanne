#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests is in the same directory as this file, as is the config file (**WRF.cfg**)
    * Test are run e.g. by::

        python -m WRF

    * For now, import troubles (can't import from higher level than package) are circumvented with sys.path.append().

.. TODO::

    * interpolation by precomputed spatial matrix

"""
import re, sys, unittest
sys.path.append('..')
from glob import glob
import os.path as pa
import xarray as xr
import pandas as pd
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import datetime, timedelta
from configparser import ConfigParser
from pyproj import Proj
from scipy import interpolate as ip
from geo import proj_params, affine
import helpers as hh


class OUT(object):
    """WRFOUT file concatenator, for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).
    Class variable *max_workers* controls how many threads are used.

    :param paths: list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
    :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.

    """
    max_workers = 16
    config_file = './WRF.cfg'

    def __init__(self, paths=None, hour=None):
        dirs = []
        if paths is None:
            self.config = ConfigParser()
            self.config.read(self.config_file)
            paths = [p for p in self.config['wrfout'].values() if pa.isdir(p)]
        for p in paths:
            dirs.extend([d for d in sorted(glob(pa.join(p, 'c01_*'))) if pa.isdir(d)])
        self.dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        print('WRF.OUT initialized with {} directories'.format(len(self.dirs)))

    def run(self, outfile, **kwargs):
        i = 0
        while len(self.dirs) > 0:
            try:
                self.concat(**kwargs)
            except:
                self.data.to_netcdf('{}{}.nc'.format(outfile, i))
                t = self.data.indexes['start'] - pd.Timedelta(kwargs['dt'], 'h')
                s = [d.strftime('%Y%m%d%H') for d in t]
                self.dirs = [d for d in self.dirs if d[-10:] not in s]
                del self.data
                i += 1

    def concat(self, var, domain, stations=None, lead_day=None, hour=None, from_date=None, dt=-4, prefix='wrfout'):
        """Concatenate the found WRFOUT files. If ``stations`` is given as a DataFrame as returned by :meth:`.CEAZA.Downloader.get_stations`, the variables are spatially interpolated to the lat/lon values in that DataFrame. If ``stations`` is set to any expression that evaluates to ``True``, the default stations data is loaded as given in a config file (class variable :attr:`.config_file`). If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :param var: Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
        :param domain: Domain specifier for which to search among WRFOUT-files.
        :param lead_day: Lead day of the forecast for which to search, if only one particular lead day is desired.
        :param stations: 'stations' DataFrame with locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable(s) is (are) to be interpolated, or expression which evaluates to ``True`` if the default station data is to be used (as saved in **WRF.cfg**).
        :param from_date: From what date onwards to search, if a start date is desired.
        :param dt: Time difference to UTC in hours by which to shift the time index
        :param prefix: Prefix of the WRFOUT files (default **wrfout**).
        :returns: concatenated data object
        :rtype: :class:`~xarray.Dataset`

        """
        start = timer()
        glob_pattern = '{}_{}_*'.format(prefix, domain)

        if stations is True:
            with pd.HDFStore(self.config['stations']['sta']) as sta:
                stations = sta['stations']

        func = partial(self._extract, var, glob_pattern, lead_day, stations, dt)

        self.data = func(self.dirs[0])
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
            for i, f in enumerate(exe.map(func, self.dirs[1:])):
                self.data = xr.concat((self.data, f), 'start' if lead_day is None else 'Time')
        self.data = self.data.sortby('start' if lead_day is None else 'XTIME')
        print('Time taken: {:.2f}'.format(timer() - start))

    @staticmethod
    def _extract(var, glob_pattern, lead_day, stations, dt, d):
        with xr.open_mfdataset(pa.join(d, glob_pattern)) as ds:
            print('using: {}'.format(ds.START_DATE))
            x = ds[np.array([var]).flatten()].to_array().sortby('Time') # this seems to be at the root of Dask warnings
            x['XTIME'] = x.XTIME + np.timedelta64(dt, 'h')
            if lead_day is not None:
                t = x.XTIME.to_index()
                x = x.isel(Time = (t - t.min()).days == lead_day)
            if stations is not None:
                x = xr_interp(x, proj_params(ds), stations)
            x.load()
            x.expand_dims('start')
            x['start'] = x.XTIME.min()
            return x.load().to_dataset('variable')

def grid_interp(xy, data, ij, method='linear'):
    "Interpolation for data on a sufficiently regular mesh."
    m, n = data.shape[:2]
    tg = affine(*xy)
    coords = np.roll(tg(np.r_['0,2', ij]).T, 1, 1)
    return [ip.interpn((range(m), range(n)), data[:, :, k], coords, method, bounds_error=False)
         for k in range(data.shape[2])]

def xr_interp(v, proj_params, stations, method='linear'):
    p = Proj(**proj_params)
    ij = p(*stations.loc[:, ('lon', 'lat')].as_matrix().T)
    xy = p(hh.g2d(v['XLONG']), hh.g2d(v['XLAT']))
    x = v.stack(n = v.dims[:-2]).transpose(*v.dims[-2:], 'n')
    y = grid_interp(xy, x.values, ij, method=method)
    ds = xr.DataArray(y, coords=[('n', x.indexes['n']), ('station', stations.index)]).unstack('n')
    ds.coords['XTIME'] = ('Time', v.XTIME)
    return ds


class ConcatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrf = OUT(hour=0)
        cls.wrf.dirs = cls.wrf.dirs[:3]

    def test_lead_day_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_intp']) as data:
            self.wrf.concat('T2', 'd03', True, 1, 0)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])

    def test_lead_day_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_whole']) as data:
            self.wrf.concat('T2', 'd03', lead_day=1, hour=0)
            # I can't get the xarray.testing method of the same name to work (fails due to timestamps)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])

    def test_all_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['all_whole']) as data:
            self.wrf.concat('T2', 'd03')
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])

    def test_all_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['all_intp']) as data:
            self.wrf.concat('T2', 'd03', True)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])


class lead(object):
    max_workers = 16
    def __init__(self, paths, domain, prefix='wrfout'):
        self.dirs = []
        self._s = '{}_{}_*'.format(prefix, domain)
        for p in paths:
            self.dirs.extend(sorted(glob(pa.join(p, 'c01_*'))))
        self.dirs = [d for d in self.dirs if pa.isdir(d)]
        self.files = [glob(pa.join(d, self._s)) for d in self.dirs]

    def remove_from_list(self, nc_var):
        self.dirs = [d for d in self.dirs
                if np.datetime64(datetime.strptime(re.search('c01_(.+)', d).group(1), '%Y%m%d%H'))
                  not in nc_var.start.values]

    def set_dirs(self, dirs):
        self.dirs = [d for d in dirs if pa.isdir(d)]


    def concat(self, var, path, dt=-4):
        sim = partial(self.by_sim, var, self._s, dt)
        x = sim(self.dirs.pop(0))[1]
        self.x = xr.concat((self.x, x), 'start') if hasattr(self, 'x') else x
        with ThreadPoolExecutor(max_workers = self.max_workers) as exe:
            for i, r in enumerate(exe.map(sim, self.dirs)):
                d, x = r
                self.x = xr.concat((self.x, x), 'start')
                self.dirs.remove(d)
                if (i + 1) % 50 == 0:
                    self.x.to_netcdf(path)
        self.x.reindex(start = self.x.start[self.x.start.argsort()]).to_netcdf(path)

    @staticmethod
    def by_sim(var, s, dt, d):
        with xr.open_mfdataset(pa.join(d, s)) as ds:
            ds['XTIME'] = ds.XTIME + np.timedelta64(dt, 'h')
            x = ds[var]
            x = x.reindex(Time = np.argsort(x.XTIME.load()))
            print('using: {}'.format(ds.START_DATE))
            # x.coords['timestep'] = ('Time', np.arange(len(x.Time)))
            # x.swap_dims({'Time': 'timestep'})
            x.expand_dims('start')
            x['start'] = x.XTIME.values.min()
            return d, x.load()


def run_lead(paths, var):
    l = lead(paths, 'd03')
    i = 0
    while len(l.dirs) > 0:
        fn = '{}{}.nc'.format(var, i)
        try:
            l.concat(var, fn)
        except:
            l.x.to_netcdf(fn)
            del l.x
            i += 1
            continue

if __name__ == '__main__':
    unittest.main()
