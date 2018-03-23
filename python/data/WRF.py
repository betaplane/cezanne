#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests is in the same directory as this file, but the config file (**tests.cfg**) is in ~/Documents/code
    * This directory is also from where the tests have to be run, e.g.::

        python -m python.data.WRF

.. TODO::

    * remove the ``hour`` keyword

"""
from glob import glob
import os.path as pa
import xarray as xr
import pandas as pd
import numpy as np
import re
from functools import partial
from ..interpolation import xr_interp
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import datetime, timedelta
import unittest
from configparser import ConfigParser


class OUT(object):
    """WRFOUT file concatenator, for a specifc forecast lead day only (I believe), and with (optional) interpolation to station locations (see :meth:`.netcdf`). Class variable *max_workers* controls how many threads are used.

    :param paths: list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
    :param domain: domain specifier for which to search among WRFOUT-files
    :param lead_day: lead day of the forecast for which to search
    :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC)
    :param from_date: from what date onwards to search

    """
    max_workers = 16

    def __init__(self, paths):
        self.dirs = []
        for p in paths:
            self.dirs.extend([d for d in sorted(glob(pa.join(p, 'c01_*'))) if pa.isdir(d)])
        print('WRF.OUT initialized with {} directories'.format(len(self.dirs)))

    def get_files(self, domain, lead_day, hour, from_date=None, prefix='wrfout'):
        name = partial(self._name, domain, lead_day, prefix)
        if from_date is None:
            files = [name(d) for d in self.dirs if d[-2:] == '{:02}'.format(hour)]
        else:
            dh = (from_date - timedelta(days=lead_day)).strftime('%Y%m%d%H')
            files = [name(d) for d in self.dirs if d[-10:] >= dh]
        return [f for f in files if pa.isfile(f)]

    @staticmethod
    def _name(dom, lead, prefix, d):
        dt = datetime.strptime(pa.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(pa.join(d, '*_{}_*'.format(dom)))
        s = (dt + timedelta(days=lead)).strftime('%Y-%m-%d_%H:%M:%S')
        return pa.join(d, '{}_{}_{}'.format(prefix, dom, s))

    @staticmethod
    def _extract(var, dt, fp):
        with xr.open_dataset(fp) as ds:
            out = xr.merge([ds[v].load() for v in var])
            out['XTIME'] = out['XTIME'] + pd.Timedelta(dt, 'h')
            return out

    def concat(self, var, domain, stations=None, lead_day=None, hour=None, from_date=None, dt=-4, prefix='wrfout'):
        """Concatenate the found WRFOUT files.

        :param var: Name of variable to extract. Can be an :obj:`iterable` if ``stations=None`` (in which case several variables can be extracted at the same time).
        :param stations: 'stations' DataFrame with locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable is to be interpolated
        :param dt: time difference to UTC in hours by which to shift the time index
        :returns: concatenated data object
        :rtype: pandas.DataFrame if *stations* is given, xarray.DataFrame otherwise

        """
        start = timer()
        if lead_day is None:
            glob_pattern = '{}_{}_*'.format(prefix, domain)
            func = partial(self.by_sim, var, glob_pattern, dt)
            iter = self.dirs
        else:
            try:
                iter = self.files
            except:
                iter = self.select_files(domain, lead_day, hour, from_date, prefix)
                self.files = iter

            if stations is None:
                func = partial(self._extract, np.asarray([var]).flatten(), dt)
            else:
                func = partial(xr_interp, var=var, stations=stations, dt=dt)

        dim = 'start' if lead_day is None else 'Time'
        ds = func(iter[0])
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            for i, f in enumerate(exe.map(func, iter[1:])):
                print(iter[i])
                ds = xr.concat((ds, f), dim) if stations is None else pd.concat((ds, f), 0)
        ds = ds.sortby('Time') if stations is None else ds.sort_index()
        print('Time taken: {:.2f}'.format(timer() - start))
        return ds

    @staticmethod
    def by_sim(var, s, dt, d):
        with xr.open_mfdataset(pa.join(d, s)) as ds:
            x = ds[var].load().sortby('Time')
            x['XTIME'] = x.XTIME + np.timedelta64(dt, 'h')
            print('using: {}'.format(ds.START_DATE))
            x.expand_dims('start')
            x['start'] = x.XTIME.values.min()
            return x

class ConcatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = ConfigParser()
        config.read('tests.cfg')
        paths = [p for p in config['wrfout'].values() if pa.isdir(p)]
        with pd.HDFStore(config['stations']['sta']) as S:
            cls.sta = S['stations']
        cls.h5_data = pd.HDFStore(config['data']['h5'])
        cls.nc_data = xr.open_dataset(config['data']['nc'])
        cls.wrf = OUT(paths)
        cls.wrf.files = cls.wrf.get_files('d03', 1, 12, None, 'wrfout')[:3]

    @classmethod
    def tearDownClass(cls):
        cls.h5_data.close()
        cls.nc_data.close()

    def test_lead_day_interpolated(self):
        ds = self.wrf.concat('T2', 'd03', self.sta, 1, 12)
        pd.testing.assert_frame_equal(ds, self.h5_data['test_lead_day_interpolated'])

    def test_lead_day_whole_domain(self):
        ds = self.wrf.concat('T2', 'd03', lead_day=1, hour=12)
        # I can't get the xarray.testing method of the same name to work (fails due to timestamps)
        np.testing.assert_allclose(ds['T2'], self.nc_data['T2'])

    def test_all_whole_domain(self):
        pass

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
