#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests is in the same directory as this file, as is the config file (**WRF.cfg**)
    * Test are run e.g. by::

        python -m WRF

.. TODO::

    * interpolation by precomputed spatial matrix

"""
import re, unittest
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


class OUT(object):
    """WRFOUT file concatenator, for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).
    Class variable *max_workers* controls how many threads are used.

    :param paths: list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
    :param domain: Domain specifier for which to search among WRFOUT-files.
    :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.
    :param from_date: From what date onwards to search (only simulation start dates), if a start date is desired.
    :type from_date: %Y%m%d :obj:`str`
    :param stations: DataFrame with station locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable(s) is (are) to be interpolated, or expression which evaluates to ``True`` if the default station data is to be used (as saved in **WRF.cfg**).
    :param interpolator: Which interpolator (if any) to use: ``scipy`` - use :class:`python.intp.GridInterpolator`; ``intp`` - use :class:`python.intp.BilinearInterpolator`.

    """
    max_workers = 16
    config_file = './WRF.cfg'

    def __init__(self, paths=None, domain=None, hour=None, from_date=None, stations=None, interpolator=None, prefix='wrfout'):
        dirs = []
        self._glob_pattern = '{}_{}_*'.format(prefix, domain)
        if paths is None:
            self.config = ConfigParser()
            self.config.read(self.config_file)
            paths = [p for p in self.config['wrfout'].values() if pa.isdir(p)]
        for p in paths:
            dirs.extend([d for d in sorted(glob(pa.join(p, 'c01_*'))) if pa.isdir(d)])
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        self.dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            f = glob(pa.join(self.dirs[0], self._glob_pattern))
            if interpolator == 'scipy':
                from interpolate import GridInterpolator
                with xr.open_dataset(f[0]) as ds:
                    self.intp = GridInterpolator(ds, self.stations)
            elif interpolator == 'intp':
                from interpolate import BilinearInterpolator
                with xr.open_dataset(f[0]) as ds:
                    self.intp = BilinearInterpolator(ds, self.stations)

        print('WRF.OUT initialized with {} directories'.format(len(self.dirs)))

    @property
    def stations(self):
        try:
            return self._stations
        except:
            with pd.HDFStore(self.config['stations']['sta']) as sta:
                self._stations = sta['stations']
            return self._stations

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

    def concat(self, var, interpolate=None, lead_day=None, dt=-4):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.OUT` or read in from the :class:`~pandas.HDFStore` specified in the :attr:`.config_file`. If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :param var: Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
        :param interpolate: Whether or not to interpolate to station locations (see :class:`.OUT`).
        :type interpolated: :obj:`bool`
        :param lead_day: Lead day of the forecast for which to search, if only one particular lead day is desired.
        :param dt: Time difference to UTC *in hours* by which to shift the time index.
        :type dt: :obj:`int`
        :returns: concatenated data object
        :rtype: :class:`~xarray.Dataset`

        """
        start = timer()

        func = partial(self._extract, var, self._glob_pattern, lead_day, dt,
                       self.intp if interpolate else None)

        self.data = func(self.dirs[0])
        if len(self.dirs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
                for i, f in enumerate(exe.map(func, self.dirs[1:])):
                    self.data = xr.concat((self.data, f), 'start' if lead_day is None else 'Time')
            self.data = self.data.sortby('start' if lead_day is None else 'XTIME')
        print('Time taken: {:.2f}'.format(timer() - start))

    @staticmethod
    def _extract(var, glob_pattern, lead_day, dt, interp, d):
        with xr.open_mfdataset(pa.join(d, glob_pattern)) as ds:
            print('using: {}'.format(ds.START_DATE))
            x = ds[np.array([var]).flatten()].to_array().sortby('XTIME') # this seems to be at the root of Dask warnings
            x['XTIME'] = x.XTIME + np.timedelta64(dt, 'h')
            if lead_day is not None:
                t = x.XTIME.to_index()
                x = x.isel(Time = (t - t.min()).days == lead_day)
            if interp is not None:
                x = interp(x)
            x.load()
            x.expand_dims('start')
            x['start'] = x.XTIME.min()
            return x.load().to_dataset('variable')


class ConcatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrf = OUT(domain='d03', hour=0, interpolator='intp')
        cls.wrf.dirs = cls.wrf.dirs[:3]

    def test_lead_day_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_intp']) as data:
            self.wrf.concat('T2', True, 1)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'], rtol=1e-4)

    def test_lead_day_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_whole']) as data:
            self.wrf.concat('T2', lead_day=1)
            # I can't get the xarray.testing method of the same name to work (fails due to timestamps)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])

    def test_all_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['all_whole']) as data:
            self.wrf.concat('T2')
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'])

    def test_all_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['all_intp']) as data:
            self.wrf.concat('T2', True)
            np.testing.assert_allclose(self.wrf.data['T2'], data['T2'], rtol=1e-4)


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
