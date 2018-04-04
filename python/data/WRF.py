#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests is in the same directory as this file, as is the config file (*WRF.cfg*)
    * Test are run e.g. by::

        python -m WRF

Example Usage::

    import WRF
    w = WRF.Concatenator(domain='d03', interpolator='bilinear')
    w.run('T2-PSFC', var=['T2', 'PSFC'], interpolate=True)

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
from configparser import ConfigParser


config_file = 'WRF.cfg'
"name of the config file (in the same directory as this module)"

def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.Downloader`) with a concatenated (and interpolated to station locations) netCDF file as produced by :class:`.Concatenator`. For now, works with a single field.

    :param wrf: The DataArray or Dataset containing the concatenated and interpolated (to station locations) WRF simulations (only dimensions ``start`` and ``Time`` and coordinate ``XTIME`` are used).
    :type wrf: :class:`~xarray.DataArray` or :class:`~xarray.Dataset`
    :param df: The DataFrame containing the station data (of the shape returned by :meth:`.CEAZA.Downloader.get_field`).
    :type df: :class:`~pandas.DataFrame`
    :returns: DataArray with ``start`` and ``Time`` dimensions aligned with **wrf**.
    :rtype: :class:`~xarray.DataArray`

    """
    xt = wrf.XTIME.astype('datetime64[h]') # necessary because some timestamps seem to be slightly off-round hours
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    cols = df.columns.get_level_values('station').intersection(wrf.station)
    return xr.DataArray(np.stack([df[c].values[idx].squeeze() for c in cols], 2),
                     coords = [wrf.coords['start'], wrf.coords['Time'], ('station', cols)])

def get_files(domain=None, lead_day=None, hour=None, from_date=None, prefix='wrfout'):
    """Function which mimics the way :class:`.Concatenator` retrieves the files to concatenate from the :data:`.config_file`, with the same keywords.

    :returns: Two arrays: one containing the list of files, one containing one tuple per directory scanned, of (directory, number of files in directory, simulation start hour)

    """
    config = ConfigParser()
    config.read(pa.join(pa.dirname(__file__), config_file))
    dirs = []
    for p in config['wrfout'].values():
        if pa.isdir(p):
            dirs.extend([d for d in sorted(glob(pa.join(p, 'c01_*'))) if pa.isdir(d)])
    dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
    dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]

    def name(d):
        ts = pd.Timestamp.strptime(pa.split(d)[1], 'c01_%Y%m%d%H')
        if lead_day is None:
            g = glob(pa.join(d, '{}_{}_*'.format(prefix, domain)))
        else:
            s = (ts + pd.Timedelta(lead_day, 'D')).strftime('%Y-%m-%d_%H:%M:%S')
            g = [f for f in glob(pa.join(d, '{}_{}_{}'.format(prefix, domain, s))) if pa.isfile(f)]
        return (g, len(g), ts.hour)

    files, length, hour = zip(*[name(d) for d in dirs])
    return [f for d in files for f in d], list(zip(dirs, length, hour))


class Concatenator(object):
    """WRFOUT file concatenator, for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).
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


    def __init__(self, domain, paths=None, hour=None, from_date=None, stations=None, interpolator='scipy', prefix='wrfout'):
        dirs = []
        self._glob_pattern = '{}_{}_*'.format(prefix, domain)
        if paths is None:
            self.config = ConfigParser()
            assert len(self.config.read(pa.join(pa.dirname(__file__), config_file))) > 0, \
                "config file not readable"
            paths = [p for p in self.config['wrfout'].values() if pa.isdir(p)]
        for p in paths:
            dirs.extend([d for d in sorted(glob(pa.join(p, 'c01_*'))) if pa.isdir(d)])
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        self.dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]

        assert len(self.dirs) > 0, "no directories added"

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            f = glob(pa.join(self.dirs[0], self._glob_pattern))
            if interpolator == 'scipy':
                from .interpolate import GridInterpolator
                with xr.open_dataset(f[0]) as ds:
                    self.intp = GridInterpolator(ds, self.stations)
            elif interpolator == 'bilinear':
                from .interpolate import BilinearInterpolator
                with xr.open_dataset(f[0]) as ds:
                    self.intp = BilinearInterpolator(ds, self.stations)

        print('WRF.Concatenator initialized with {} directories'.format(len(self.dirs)))

    @property
    def stations(self):
        try:
            return self._stations
        except:
            with pd.HDFStore(self.config['stations']['sta']) as sta:
                self._stations = sta['stations']
            return self._stations

    def run(self, outfile, **kwargs):
        """Wrapper around the :meth:`.concat` call which writes out a netCDF file whenever an error occurs and restarts with all already processed directories removed from :attr:`.dirs`

        :param outfile: base name of the output netCDF file (no extension, numberings are added in case of restarts)
        :Keyword arguments:

            Are all passed to :meth:`.concat`

        """
        i = 0
        def write(filename, start):
            global i
            self.data.to_netcdf(filename)
            t = self.data.indexes['start'] - pd.Timedelta(self.dt, 'h')
            s = [d.strftime('%Y%m%d%H') for d in t]
            self.dirs = [d for d in self.dirs if d[-10:] not in s]
            with open('timing.txt') as f:
                f.write('{} dirs in {} seconds, file {}'.format(
                    len_dirs - len(self.dirs), timer() - start), filename)
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


    def concat(self, var, interpolate=None, lead_day=None, dt=-4):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config_file`. If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :param var: Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
        :param interpolate: Whether or not to interpolate to station locations (see :class:`.Concatenator`).
        :type interpolated: :obj:`bool`
        :param lead_day: Lead day of the forecast for which to search, if only one particular lead day is desired.
        :param dt: Time difference to UTC *in hours* by which to shift the time index.
        :type dt: :obj:`int`
        :returns: concatenated data object
        :rtype: :class:`~xarray.Dataset`

        """
        start = timer()
        self.dt = dt

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
        cls.wrf = Concatenator(domain='d03', hour=0, interpolator='bilinear')
        cls.wrf.dirs = cls.wrf.dirs[:3]

    def test_lead_day_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_day1']) as data:
            self.wrf.concat('T2', True, 1)
            np.testing.assert_allclose(self.wrf.data['T2'], data['interp'], rtol=1e-4)

    def test_lead_day_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['lead_day1']) as data:
            self.wrf.concat('T2', lead_day=1)
            # I can't get the xarray.testing method of the same name to work (fails due to timestamps)
            np.testing.assert_allclose(self.wrf.data['T2'], data['field'])

    def test_all_whole_domain(self):
        with xr.open_dataset(self.wrf.config['tests']['all_days']) as data:
            self.wrf.concat('T2')
            np.testing.assert_allclose(self.wrf.data['T2'], data['field'])

    def test_all_interpolated(self):
        with xr.open_dataset(self.wrf.config['tests']['all_days']) as data:
            self.wrf.concat('T2', True)
            np.testing.assert_allclose(self.wrf.data['T2'], data['interp'], rtol=1e-4)



if __name__ == '__main__':
    unittest.main()
