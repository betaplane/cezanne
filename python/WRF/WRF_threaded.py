#!/usr/bin/env python
"""
WRFOUT concatenation (xarray version)
-------------------------------------

Example Usage::

    from data import WRF
    w = WRF.Concatenator(domain='d03', interpolator='bilinear')
    w.start('T2-PSFC', var=['T2', 'PSFC'], interpolate=True)

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
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from . import *


class Concatenator(WRFiles):
    """WRFOUT file concatenator (xarray version), for a specifc forecast lead day or for all data arranged in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).
    Class variable *max_workers* controls how many threads are used.

    :param domain: Domain specifier for which to search among WRFOUT-files.
    :type from_date: %Y%m%d :obj:`str`
    :param stations: DataFrame with station locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable(s) is (are) to be interpolated, or expression which evaluates to ``True`` if the default station data is to be used (as saved in :data:`config_file`).
    :param interpolator: Which interpolator (if any) to use: ``scipy`` - use :class:`~.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~.interpolate.BilinearInterpolator`.

    :Keyword arguments:
        * **paths** - list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
        * **hour** - hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.
        * **from_date** - From what date onwards to search (only simulation start dates), if a start date is desired.

    """
    max_workers = Integer(16, help="number of threads to be used").tag(config=True)
    "number of threads to be used"

    def __init__(self, domain, *args, stations=None, interpolator='scipy', dt=-4, **kwargs):
        super().__init__(*args, **kwargs)

        self.dt = dt
        self._glob_pattern = '{}_{}_*'.format(self.wrfout_prefix, domain)

        assert len(self.dirs) > 0, "no directories added"

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            f = glob(os.path.join(self.dirs[0], self._glob_pattern))
            with xr.open_dataset(f[0]) as ds:
                self.intp = getattr(import_module('data.interpolate'), {
                    'scipy': 'GridInterpolator',
                    'bilinear': 'BilinearInterpolator'
                }[interpolator])(ds, stations = self.stations)

        print('Concatenator initialized with {} directories, interpolator {}'.format(len(self.dirs), interpolator))

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - pd.Timedelta(self.dt, 'h')
        s = [d.strftime('%Y%m%d%H') for d in t]
        return [d for d in self.dirs if d[-10:] not in s]

    def start(self, outfile, previous_file=None, **kwargs):
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
        with xr.open_mfdataset(os.path.join(d, glob_pattern)) as ds:
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
