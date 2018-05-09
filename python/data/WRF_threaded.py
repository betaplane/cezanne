#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

Example Usage::

    from data import WRF
    w = WRF.Concatenator(domain='d03', interpolator='bilinear')
    w.run('T2-PSFC', var=['T2', 'PSFC'], interpolate=True)

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests is in the same directory as this file
    * Test are run e.g. by::

        python -m unittest data.WRF.Tests

.. warning::

    The current layout with ThreadPoolExecutor seems to work on the UV only if one sets::

        export OMP_NUM_THREADS=1

    (`a conflict between OpenMP and dask? <https://stackoverflow.com/questions/39422092/error-with-omp-num-threads-when-using-dask-distributed>`_)


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
# from importlib.util import spec_from_file_location, module_from_spec
from importlib import import_module
from os.path import join, dirname
from . import config


def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.Downloader`) with a concatenated (and interpolated to station locations) netCDF file as produced by :class:`.Concatenator`. For now, works with a single field.

    :param wrf: The DataArray or Dataset containing the concatenated and interpolated (to station locations) WRF simulations (only dimensions ``start`` and ``Time`` and coordinate ``XTIME`` are used).
    :type wrf: :class:`~xarray.DataArray` or :class:`~xarray.Dataset`
    :param df: The DataFrame containing the station data (of the shape returned by :meth:`.CEAZA.Downloader.get_field`).
    :type df: :class:`~pandas.DataFrame`
    :returns: DataArray with ``start`` and ``Time`` dimensions aligned with **wrf**.
    :rtype: :class:`~xarray.DataArray`

    """
    # necessary because some timestamps seem to be slightly off-round hours
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    xt = xr.DataArray(pd.Series(xt.values).dt.round('h'), coords=xt.coords).unstack('t')
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    cols = df.columns.get_level_values('station').intersection(wrf.station)
    return xr.DataArray(np.stack([np.where(idx>=0, df[c].values[idx].squeeze(), np.nan) for c in cols], 2),
                     coords = [wrf.coords['start'], wrf.coords['Time'], ('station', cols)])

class Files(object):
    """Class which mimics the way :class:`.Concatenator` retrieves the files to concatenate from the :data:`.config_file`, with the same keywords.

    """
    def __init__(self, paths=None, hour=None, from_date=None, pattern='c01_*', limit=None):
        dirs = []
        if paths is None:
            assert len(config) > 0, "config file not read"
            self.paths = [p for p in config['wrfout'].values() if pa.isdir(p)]
        for p in self.paths:
            for d in sorted(glob(pa.join(p, pattern))):
                if (pa.isdir(d) and not pa.islink(d)):
                    dirs.append(d)
                    if limit is not None and len(dirs) == limit:
                        break
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        self.dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]

    @staticmethod
    def _name(domain, lead_day, prefix, d):
        ts = pd.Timestamp.strptime(pa.split(d)[1][-10:], '%Y%m%d%H')
        if lead_day is None:
            g = glob(pa.join(d, '{}_{}_*'.format(prefix, domain)))
        else:
            s = (ts + pd.Timedelta(lead_day, 'D')).strftime('%Y-%m-%d_%H:%M:%S')
            g = [f for f in glob(pa.join(d, '{}_{}_{}'.format(prefix, domain, s))) if pa.isfile(f)]
        return (g, len(g), ts.hour)

    def files(self, domain=None, lead_day=None, prefix='wrfout'):
        name = partial(self._name, domain, lead_day, prefix)
        self.files, self.length, self.hour = zip(*[name(d) for d in self.dirs])

    def by_sim_length(self, n):
        return [f for d in np.array(self.files)[np.array(self.length) == n] for f in d]

    @classmethod
    def first(cls, domain, lead_day=None, hour=None, from_date=None, pattern='c01_*', prefix='wrfout', opened=True):
        """Get the first netCDF file matching the given arguments (see :class:`Concatenator` for a description), based on the configuration values (section *wrfout*) in the global config file.

        """
        f = cls(hour=hour, from_date=from_date, pattern=pattern, limit=1)
        name = partial(cls._name, domain, lead_day, prefix)
        files, _, _ = name(f.dirs[0])
        return xr.open_dataset(files[0]) if opened else files[0]

    @staticmethod
    def stations():
        with pd.HDFStore(config['stations']['sta']) as S:
            return S['stations']


class Concatenator(object):
    """WRFOUT file concatenator (xarray version), for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).
    Class variable *max_workers* controls how many threads are used.

    :param domain: Domain specifier for which to search among WRFOUT-files.
    :param paths: list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
    :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.
    :param from_date: From what date onwards to search (only simulation start dates), if a start date is desired.
    :type from_date: %Y%m%d :obj:`str`
    :param stations: DataFrame with station locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable(s) is (are) to be interpolated, or expression which evaluates to ``True`` if the default station data is to be used (as saved in :data:`config_file`).
    :param interpolator: Which interpolator (if any) to use: ``scipy`` - use :class:`~.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~.interpolate.BilinearInterpolator`.

    """
    max_workers = 16
    "number of threads to be used"


    def __init__(self, domain, paths=None, hour=None, from_date=None, stations=None, interpolator='scipy', prefix='wrfout', dt=-4):
        self.dt = dt
        self._glob_pattern = '{}_{}_*'.format(prefix, domain)
        files = Files(paths, hour, from_date)
        self.dirs = files.dirs

        assert len(self.dirs) > 0, "no directories added"

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            f = glob(pa.join(self.dirs[0], self._glob_pattern))
            with xr.open_dataset(f[0]) as ds:
                self.intp = getattr(import_module('data.interpolate'), {
                    'scipy': 'GridInterpolator',
                    'bilinear': 'BilinearInterpolator'
                }[interpolator])(ds, stations = self.stations)

        print('WRF.Concatenator initialized with {} directories'.format(len(self.dirs)))

    @property
    def stations(self):
        try:
            return self._stations
        except:
            with pd.HDFStore(config['stations']['sta']) as sta:
                self._stations = sta['stations']
            return self._stations

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - pd.Timedelta(self.dt, 'h')
        s = [d.strftime('%Y%m%d%H') for d in t]
        return [d for d in self.dirs if d[-10:] not in s]

    def run(self, outfile, previous_file=None, **kwargs):
        """Wrapper around the :meth:`.concat` call which writes out a netCDF file whenever an error occurs and restarts with all already processed directories removed from :attr:`.dirs`

        :param outfile: base name of the output netCDF file (no extension, numberings are added in case of restarts)
        :Keyword arguments:

            Are all passed to :meth:`.concat`

        """
        if previous_file is not None:
            with xr.open_dataset(previous_file) as ds:
                self.dirs = self.remove_dirs(ds)

        i = 0
        def write(filename, start):
            global i
            self.data.to_netcdf(filename)
            self.dirs = self.remove_dirs(self.data)
            with open('timing.txt', 'a') as f:
                f.write('{} dirs in {} seconds, file {}'.format(
                    len_dirs - len(self.dirs), timer() - start, filename))
            i += 1

        while len(self.dirs) > 0:
            try:
                start = timer()
                len_dirs = len(self.dirs)
                self.concat(**kwargs)
            except:
                write('{}{}.nc'.format(outfile, i), start)
            finally:
                write('{}{}.nc'.format(outfile, i), start)


    def concat(self, variables, interpolate=None, lead_day=None, func=None):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config`. If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :param var: Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
        :param interpolate: Whether or not to interpolate to station locations (see :class:`.Concatenator`).
        :type interpolated: :obj:`bool`
        :param lead_day: Lead day of the forecast for which to search, if only one particular lead day is desired.
        :param func: callable to be applied to the data before concatenation (after interpolation)
        :type func: :obj:`callable`
        :param dt: Time difference to UTC *in hours* by which to shift the time index.
        :type dt: :obj:`int`
        :returns: concatenated data object
        :rtype: :class:`~xarray.Dataset`

        """
        start = timer()

        func = partial(self._extract, variables, self._glob_pattern, lead_day, self.dt,
                       self.intp if interpolate else None, func)

        self.data = func(self.dirs[0])
        if len(self.dirs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
                for i, f in enumerate(exe.map(func, self.dirs[1:])):
                    self.data = xr.concat((self.data, f), 'start')
            if lead_day is None:
                self.data = self.data.sortby('start')
            else:
                self.data = self.data.rename({'Time': 't'}).stack(Time=('start', 't')).sortby('XTIME')
        print('Time taken: {:.2f}'.format(timer() - start))

    @staticmethod
    def _extract(var, glob_pattern, lead_day, dt, interp, func, d):
        with xr.open_mfdataset(pa.join(d, glob_pattern)) as ds:
            print('using: {}'.format(ds.START_DATE))
            x = ds[np.array([var]).flatten()].sortby('XTIME') # this seems to be at the root of Dask warnings
            x['XTIME'] = x.XTIME + np.timedelta64(dt, 'h')
            if lead_day is not None:
                t = x.XTIME.to_index()
                x = x.isel(Time = (t - t.min()).days == lead_day)
            if interp is not None:
                x = x.apply(interp.xarray)
            if func is not None:
                x = func(x)
            x.load()
            for v in x.data_vars:
                x[v] = x[v].expand_dims('start')
            x['start'] = ('start', pd.DatetimeIndex([x.XTIME.min().item()]))
            return x


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrf = Concatenator(domain='d03', interpolator='bilinear')
        cls.wrf.dirs = [d for d in cls.wrf.dirs if re.search('c01_2016120[1-3]', d)]

    def test_lead_day_interpolated(self):
        with xr.open_dataset(config['tests']['lead_day1']) as data:
            self.wrf.concat('T2', True, 1)
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('station', 'time'),
                data['interp'].transpose('station', 'time'), rtol=1e-4)

    def test_lead_day_whole_domain(self):
        with xr.open_dataset(config['tests']['lead_day1']) as data:
            self.wrf.concat('T2', lead_day=1)
            # I can't get the xarray.testing method of the same name to work (fails due to timestamps)
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('south_north', 'west_east', 'time'),
                data['field'].transpose('south_north', 'west_east', 'time'))

    def test_all_whole_domain(self):
        with xr.open_dataset(config['tests']['all_days']) as data:
            self.wrf.concat('T2')
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('start', 'Time', 'south_north', 'west_east'),
                data['field'].transpose('start', 'Time', 'south_north', 'west_east'))

    def test_all_interpolated(self):
        with xr.open_dataset(config['tests']['all_days']) as data:
            self.wrf.concat('T2', True)
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('start', 'station', 'Time'),
                data['interp'].transpose('start', 'station', 'Time'), rtol=1e-3)

def run_tests():
    suite = unittest.TestSuite()
    suite.addTests([Tests(t) for t in Tests.__dict__.keys() if t[:4]=='test'])
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    unittest.main()
