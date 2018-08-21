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

        python -m unittest WRF.tests
    * The :class:`Concatenator` class uses the following attributes and methods from its super:
        * dirs
        * stations / _stations
        * wrfout_prefix

.. warning::

    The current layout with ThreadPoolExecutor seems to work on the UV only if one sets::

        export OMP_NUM_THREADS=1

    (`a conflict between OpenMP and dask? <https://stackoverflow.com/questions/39422092/error-with-omp-num-threads-when-using-dask-distributed>`_)

.. TODO::

    * better file writing (regularly instead of in case of error) in :meth:`start`

"""
import re
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import timedelta
from WRF import *


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

    interpolator = Unicode('scipy').tag(config=True)

    outfile = Unicode('out.nc').tag(config=True)
    prev_file = Unicode().tag(config=True)

    variables = List().tag(config=True)
    domain = Unicode('d03').tag(config=True)
    interpolate = Bool(False).tag(config=True)
    utc_delta = Instance(timedelta, kw={'hours': -4})
    lead_day = Integer(-1).tag(config=True)
    # func = Instance(callable).tag(config=True)

    aliases = {'d': 'Concatenator.domain',
               'o': 'Concatenator.outfile',
               'v': 'Concatenator.variables',
               'p': 'Concatenator.prev_file'}

    flags = {'i': ({'Concatenator': {'interpolate': True}}, "interpolate to station locations")}

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - self.utc_delta # subtraction because we're going in the reverse direction here
        s = [d.strftime('%Y%m%d%H') for d in t]  # (from modified time in previous_file to original WRFOUT)
        n = len(self.dirs)
        self.dirs = [d for d in self.dirs if d[-10:] not in s]
        self.log.info("%s directories removed.", n - len(self.dirs))

    def start(self, **kwargs):
        """Wrapper around the :meth:`.concat` call which writes out a netCDF file whenever an error occurs and restarts with all already processed directories removed from :attr:`.dirs`

        :Keyword arguments:
            * **outfile** - Base name of the output netCDF file (no extension, numberings are added in case of restarts, in the form '_part#' where # is the number). Defaults to the :attr:`outfile` trait which can be set by command line ('-o' flag) and config.

            Any other keyword argument is passed to :meth:`.concat`.

        """
        fileno = 0
        # check if there is a file with self.outfile and _part (i.e. iterruptus)
        g = sorted(glob('{}_part*nc'.format(self.outfile)))
        if len(g) > 0:
            self.log.info("Restarting from file %s.", g[-1])
            fileno = int(re.search('_part(\d+)', g[-1]).group(1)) + 1
            for f in g:
                with xr.open_dataset(f) as ds:
                    self.remove_dirs(ds)

        def write(filename, start):
            self.data.to_netcdf(filename)
            len_dirs = len(self.dirs)
            self.remove_dirs(self.data)
            with open('timing.txt', 'a') as f:
                f.write('{} dirs in {} seconds, file {}\n'.format(
                    len_dirs - len(self.dirs), timer() - start, filename))

        while len(self.dirs) > 0:
            try:
                start = timer()
                self.concat(**kwargs)
                write('{}_final.nc'.format(self.outfile), start)
            except:
                write('{}_part{}.nc'.format(self.outfile, fileno), start)
                fileno += 1

    def concat(self, **kwargs):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config`. If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :Keyword Arguments:

            (Any of the :mod:`traitlets.config`-configurable traits can be overridden by keyword arguments.)

            * **variables** - Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
            * **domain** (:obj:`str`) - Which of the WRF domains (e.g. ``d03``) to use.
            * **interpolate** (obj:`bool`) - Whether or not to interpolate to station locations (see :class:`.Concatenator`).
            * **lead_day** - (:obj:`int`) Lead day of the forecast for which to search, if only one particular lead day is desired. (``-1`` denotes no particular lead day.)
            * **func** (:obj:`callable`) - callable to be applied to the data before concatenation (after interpolation)

        :returns: concatenated data object
        :rtype: :class:`~xarray.Dataset`

        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._wrfout_glob = '{}_{}_*'.format(self.wrfout_prefix, self.domain)

        if self.interpolate:
            if not hasattr(self, '_interpolator'):
                f = glob(os.path.join(self.dirs[0], self._wrfout_glob))
                with xr.open_dataset(f[0]) as ds:
                    intp = getattr(import_module('data.interpolate'), {
                        'scipy': 'GridInterpolator',
                        'bilinear': 'BilinearInterpolator'
                    }[self.interpolator])(ds, stations = self.stations)
                self._interpolator = intp
            self.log.info("Interpolation requested with interpolator %s", self.interpolator)

        start = timer()
        self.data = func(self.dirs[0])
        if len(self.dirs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
                for i, f in enumerate(exe.map(self._extract, self.dirs[1:])):
                    self.data = xr.concat((self.data, f), 'start')
            if lead_day == -1:
                self.data = self.data.sortby('start')
            else:
                self.data = self.data.rename({'Time': 't'}).stack(Time=('start', 't')).sortby('XTIME')
        print('Time taken: {:.2f}'.format(timer() - start))

    def _extract(self, d):
        with xr.open_mfdataset(os.path.join(d, self._wrfout_glob)) as ds:
            print('using: {}'.format(ds.START_DATE))
            x = ds[self.variables].sortby('XTIME') # this seems to be at the root of Dask warnings
            x['XTIME'] = x.XTIME + pd.Timedelta(utc_delta)
            if self.lead_day >= 0:
                t = x.XTIME.to_index()
                x = x.isel(Time = (t - t.min()).days == lead_day)
            if self.interpolate:
                x = x.apply(self._interpolator.xarray)
            if self.func is not None: # check how that would work
                x = func(x)
            x.load()
            for v in x.data_vars:
                x[v] = x[v].expand_dims('start')
            x['start'] = ('start', pd.DatetimeIndex([x.XTIME.min().item()]))
            return x

if __name__ == '__main__':
    app = Concatenator()
    app.initialize()
    app.start()
