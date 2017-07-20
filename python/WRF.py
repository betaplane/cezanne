#!/usr/bin/env python
from glob import glob
import os.path as pa
import xarray as xr
import pandas as pd
import numpy as np
import re
from functools import partial
from interpolation import xr_interp
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import datetime, timedelta


class OUT(object):
    max_workers = 16
    def __init__(self, paths, domain, lead_day, hour, from_date=None, prefix='wrfout'):
        """Initialize WRFOUT file concatenator. Class variable *max_workers*
        controls how many threads are used.

        :param paths: list of names of base directories containing the 'c01_...' directories
        corresponding to individual forecast runs/
        :param domain: domain specifier for which to search among WRFOUT-files
        :param lead_day: lead day of the forecast for which to search
        :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC)
        :param from_date: from what date onwards to search

        """
        name = partial(self._name, domain, lead_day, prefix)
        dirs = []
        for p in paths:
            dirs.extend(sorted(glob(pa.join(p, 'c01*'))))
        if from_date is None:
            files = [name(d) for d in dirs if d[-2:] == '{:02}'.format(hour)]
        else:
            dh = (from_date - timedelta(days=lead_day)).strftime('%Y%m%d%H')
            files = [name(d) for d in dirs if d[-10:] >= dh]
        self.files = [f for f in files if pa.isfile(f)]
        print('WRF.OUT initialized with {} files'.format(len(self.files)))

    @staticmethod
    def _name(dom, lead, prefix, d):
        dt = datetime.strptime(pa.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(pa.join(d, '*_{}_*'.format(dom)))
        s = (dt + timedelta(days=lead)).strftime('%Y-%m-%d_%H:%M:%S')
        return pa.join(d, '{}_{}_{}'.format(prefix, dom, s))

    @staticmethod
    def _extract(var, fp):
        with xr.open_dataset(fp) as ds:
            return xr.merge([ds[v].load() for v in var])

    def netcdf(self, var, stations=None, dt=-4):
        """Concatenate the found WRFOUT files.

        :param var: name of variable to extract (list if stations=None)
        :param stations: pandas.DataFrame with locations to which the variables is to be interpolated
        :param dt: time difference to UTC in hours by which to shift the time index
        :returns: concatenated data object
        :rtype: pandas.DataFrame if *stations* is given, xarray.DataFrame otherwise

        """
        start = timer()
        if stations is not None:
            ds, Map = xr_interp(self.files[0], var, stations)
            intp = partial(xr_interp, var=var, stations=stations, map=Map, dt=dt)
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                self.data = {exe.submit(intp, f): f for f in self.files[1:]}
                for f in as_completed(self.data):
                    ds = pd.concat((ds, f.result()), 0)
                    print(self.data[f])
            ds.sort_index(inplace=True)
        else:
            extr = partial(self._extract, [var])
            ds = extr(self.files[0])
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                for i, f in enumerate(exe.map(extr, self.files[1:])):
                    ds = xr.concat((ds, f), 'Time')
                    print(self.files[i])
            ds['XTIME'] = ds['XTIME'] + pd.Timedelta(dt, 'h')
        print(timer() - start)
        return ds

# no timezone correction!
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


    def concat(self, var, path):
        sim = partial(self.by_sim, var, self._s)
        x = sim(self.dirs.pop(0))[1]
        self.x = xr.concat((self.x, x), 'start') if hasattr(self, 'x') else x
        with ThreadPoolExecutor(max_workers = self.max_workers) as exe:
            for i, r in enumerate(exe.map(sim, self.dirs)):
                d, x = r
                self.x = xr.concat((self.x, x), 'start')
                self.dirs.remove(d)
                if (i + 1) % 50 == 0:
                    self.x.to_netcdf(path)
        self.x.to_netcdf(path)

    @staticmethod
    def by_sim(var, s, d):
        with xr.open_mfdataset(pa.join(d, s)) as ds:
            x = ds[var]
            print('using: {}'.format(ds.START_DATE))
            x.coords['timestep'] = ('Time', np.arange(len(x.Time)))
            x.swap_dims({'Time': 'timestep'})
            x.expand_dims('start')
            x['start'] = pd.Timestamp(datetime.strptime(ds.START_DATE, '%Y-%m-%d_%H:%M:%S'))
            return d, x.load()


def run_lead(paths):
    l = lead(paths, 'd03')
    i = 0
    while len(l.dirs) > 0:
        fn = 'RAINNC{}.nc'.format(i)
        try:
            l.concat('RAINNC', fn)
        except:
            l.x.to_netcdf(fn)
            del l.x
            i += 1
            continue
