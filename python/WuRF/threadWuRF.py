#!/usr/bin/env python
__package__ = 'WuRF' # this solves the relative import from '.' when run as script
import re
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from .base import *

class CC(CCBase):
    """WRFOUT file concatenator (xarray version), for a specifc forecast lead day or for all data arranged in two temporal dimensions, and with (optional) interpolation to station location.

    If :attr:`~.CCBase.interpolate` is ``True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config`.

    If :attr:`~.CCBase.lead_day` is given, only the day's data with the given lead is taken from each daily simulation, resultng in a continuous temporal sequence. If it is not given (i.e. has a value of ``-1``), the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

    An instance of this class is configured via the :class:`traitlets.config.application.Application` interface, inherited from the :class:`.base.CCBase` class in this package. All traitlet-based class/instance attributes (they appear in code as class attributes, but will only have configured values upon instantiation) can be configured via a config file, command-line arguments (see also `CEAZAMet stations webservice`_), or keyword arguments to the ``__init__`` call. (From the command-line, help can be obtained via the flag ``--help`` or ``--help-all``.)

    Common arguments to this class and :class:`.threadWuRF.CC` are described as keyword arguments below for convenience.

    :Keyword arguments:
        * :attr:`~.threadWuRF.write_interval`
        * **paths** - List of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs.
        * **domain** - Which of the WRF domains (e.g. ``d03``) to use.
        * **hour** - Hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.
        * **from_date** - From what date onwards to search (only simulation start dates), if a start date is desired (as %Y%m%d :obj:`str`).
        * **to_date** - Up to what date to search (only simulation start dates), if an end date is desired (as %Y%m%d :obj:`str`).
        * **directory_pattern** - Glob pattern for the directory names (e.g. ``c01_*``)
        * **wrfout_prefix** - Prefix of the WRF output files (e.g. ``wrfout``).
        * **outfile** - Base name of the output netCDF file (**no extension**, numberings are added in case of restarts, in the form '_part#' where # is the number). Defaults to the :attr:`outfile` trait which can be set by command line ('-o' flag) and config.
        * **variables** - Name of variable(s) to extract.
        * **interpolator** - Which interpolator (if any) to use: ``scipy`` - use :class:`~data.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~data.interpolate.BilinearInterpolator`.
        * **interpolate** - Whether or not to interpolate to station locations (see :class:`.CC`).
        * **utc_delta** - The offset from UTC to be applied to the concatenated data (assuming the original files are in UTC).
        * **lead_day** - Lead day of the forecast for which to search, if only one particular lead day is desired. (``-1`` denotes no particular lead day.)
        * **function** - Callable to be applied to the data before concatenation (after interpolation), in dotted from ('<module>.<function>'). (**Not implemented in :mod:`.mpiWuRF` yet**)
        * **max_workers** - Maximum number of threads to use.
        * **initfile** - File containing projection information (for interpolation) and/or spatial coordinates (for netCDF4_ based concatenator).

    """
    write_interval = Integer(73).tag(config=True) # 365 / 73 = 5
    """The number of input directories to process before triggering a write-out to file (only applicable if using :meth:`start` or :meth:`concat` with argument ``interval=True``)."""

    def start(self):
        """Wrapper around the :meth:`.concat` call which writes out a netCDF file whenever an error occurs and restarts with all already processed directories removed from :attr:`.dirs`

        """
        self.fileno = 0
        r = lambda s: int(re.search('(\d+)\.nc', s).group(1))
        # check if there is a file with self.outfile and _part (i.e. iterruptus)
        g = sorted(glob('{}*nc'.format(self.outfile)), key=r)
        if len(g) > 0:
            self.log.info("Restarting from file %s.", g[-1])
            self.fileno = r(g[-1]) + 1
            for f in g:
                with xr.open_dataset(f) as ds:
                    self.remove_dirs(ds)

        while len(self.dirs) > 0:
            try:
                self.concat(True)
                self.write('final')
            except Exception as exc:
                print(exc)
                self.write('inter')

    def sort_data(self):
        if self.lead_day == -1:
            self.data = self.data.sortby('start')
        else:
            self.data = self.data.rename({'Time': 't'}).stack(Time=('start', 't')).sortby('XTIME')

    def write(self, middle):
        fn = '{}_{}_{}.nc'.format(self.outfile, middle, self.fileno)
        self.sort_data()
        self.data.to_netcdf(fn)
        n = self.remove_dirs(self.data)
        self.data = None if n==0 else self._extract2(self.dirs[0])
        self.fileno += 1

    def concat(self, interval=False):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are found from the HDF file specified as :attr:`CEAZA.Meta.file_name`.

        :param interval: Whether or not to write out files at the interval specified by :attr:`~threadWuRF.write_interval`. This is mostly intended for internal use when :meth:`concat` is called from :meth:`start`

        """
        if self.interpolate:
            if not hasattr(self, '_interpolator'):
                if self.initfile == '':
                    self.initfile = glob(os.path.join(self.dirs[0], self.file_glob))[0]
                with xr.open_dataset(self.initfile) as ds:
                    self._interpolator = getattr(import_module('data.interpolate'), {
                        'scipy': 'GridInterpolator',
                        'bilinear': 'BilinearInterpolator'
                    }[self.interpolator])(ds, stations = self.stations)
            self.log.info("Interpolation requested with interpolator %s", self.interpolator)

        start = timer()
        while self.data is None:
            self.data = self._extract2(self.dirs[0])
        if len(self.dirs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
                for i, f in enumerate(exe.map(self._extract2, self.dirs[1:])):
                    self.data = xr.concat((self.data, f), 'start')
                    if interval and ((i + 1) % self.write_interval == 0):
                        self.write('reg')
            self.sort_data()
        self.log.info('Time taken: {:.2f}'.format(timer() - start))

    def _extract(self, d):
        try:
            with xr.open_mfdataset(os.path.join(d, self.file_glob)) as ds:
                print('using: {}'.format(ds.START_DATE))
                x = ds[self.var_list].sortby('XTIME') # this seems to be at the root of Dask warnings
                x['XTIME'] = x.XTIME + pd.Timedelta(self.utc_delta)
                if self.lead_day >= 0:
                    t = x.XTIME.to_index()
                    x = x.isel(Time = (t - t.min()).days == self.lead_day)
                if self.interpolate:
                    x = x.apply(self._interpolator.xarray)
                x = self.func(x)
                x.load()
                for v in x.data_vars:
                    x[v] = x[v].expand_dims('start')
                x['start'] = ('start', pd.DatetimeIndex([x.XTIME.min().item()]))
                return x
        except IOError as ioe:
            if ioe.args[0] == 'no files to open':
                self.log.critical('Dir %s: %s', d, ioe)
                self.dirs.remove(d)

    # alternative, for the 'wrfxtrm' files which don't have 'XTIME'
    def _extract2(self, d):
        try:
            with xr.open_mfdataset(os.path.join(d, self.file_glob)) as ds:
                print('using: {}'.format(ds.START_DATE))
                t = ds['Times'].load()
                i = np.argsort(t)
                x = ds[self.var_list].isel(Time=i)
                x.coords['XTIME'] = ('Time', pd.DatetimeIndex(
                    [datetime.strptime(d.item(), '%Y-%m-%d_%H:%M:%S') for d in t.isel(Time=i).astype(str)]
                    ) + pd.Timedelta(self.utc_delta))
                if self.lead_day >= 0:
                    raise Exception('lead day extraction not implemented by this method')
                if self.interpolate:
                    x = x.apply(self._interpolator.xarray)
                x = self.func(x)
                x.load()
                for v in x.data_vars:
                    x[v] = x[v].expand_dims('start')
                x['start'] = ('start', pd.DatetimeIndex([x.XTIME.min().item()]))
                return x
        except IOError as ioe:
            if ioe.args[0] == 'no files to open':
                self.log.critical('Dir %s: %s', d, ioe)
                self.dirs.remove(d)

if __name__ == '__main__':
    # app = Concatenator(variables='T2', outfile='/HPC/arno/data/T2_new')
    app = CC()
    app.parse_command_line()
    app.start()
