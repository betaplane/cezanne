#!/usr/bin/env python
from glob import glob
import os.path as pa
import xarray as xr
import pandas as pd
from functools import partial
from interpolation import xr_interp
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import datetime, timedelta


class OUT(object):
    max_workers = 16
    def __init__(self, paths, domain, lead_day, hour, from_date=None):
        """Initialize WRFOUT file concatenator. Class variable *max_workers*
        controls how many threads are used.

        :param paths: list of names of base directories containing the 'c01_...' directories
        corresponding to individual forecast runs/
        :param domain: domain specifier for which to search among WRFOUT-files
        :param lead_day: lead day of the forecast for which to search
        :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC)
        :param from_date: from what date onwards to search

        """
        name = partial(self._name, domain,  lead_day)
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
    def _name(dom, lead, d):
        dt = datetime.strptime(pa.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(pa.join(d, '*_{}_*'.format(dom)))
        s = (dt + timedelta(days=lead)).strftime('%Y-%m-%d_%H:%M:%S')
        return pa.join(d, 'wrfout_{}_{}'.format(dom, s))

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
            df, Map = xr_interp(self.files[0], var, stations)
            intp = partial(xr_interp, var=var, stations=stations, map=Map, dt=dt)
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                self.data = {exe.submit(intp, f): f for f in self.files[1:]}
                for f in as_completed(self.data):
                    df = pd.concat((df, f.result()), 0)
                    print(self.data[f])
            df.sort_index(inplace=True)
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

