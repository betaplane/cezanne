#!/usr/bin/env python
"""
WRFOUT concatenation (xarray version)
-------------------------------------

Usage
=====

The following invocations are equivalent. From a python script / REPL::

    from data import WRF
    w = WRF.Concatenator(variables='T2', outfile='T2', domain='d03', interpolator='bilinear', interpolate=True)
    w.start()

From the command line::

    ./threads.py -v T2 -o T2 -d 'd03' --Concatenator.interpolator='bilinear' -i

`Help <https://traitlets.readthedocs.io/en/stable/config.html#subcommands>`_ can be obtained via::

    ./threads.py --help

or ``--help-all``.

Test are run e.g. by::

    python -m unittest WRF.tests

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests can be found in the directory specified by :attr:`tests.WRFTests.test_dir`.
    * The :class:`Concatenator` class uses the following attributes and methods from its super:
        * dirs
        * stations / _stations
        * file_glob / _file_glob
    * xr.concat works if Time dim has different lengths **if it has a coordinate** (right now it doesn't)
    * **opening files as xr.open_dataarray throws away the metadata of the files**

.. warning::

    The current layout with ThreadPoolExecutor seems to work on the UV only if one sets::

        export OMP_NUM_THREADS=1

    (`a conflict between OpenMP and dask? <https://stackoverflow.com/questions/39422092/error-with-omp-num-threads-when-using-dask-distributed>`_)

.. TODO::

    * better file writing (regularly instead of in case of error) in :meth:`start`
    * add coordinate to Time?

"""
import re
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from datetime import timedelta
from WRF import *

class Concatenator(WRFiles):
    """WRFOUT file concatenator (xarray version), for a specifc forecast lead day or for all data arranged in two temporal dimensions, and with (optional) interpolation to station location.

    If :attr:`lead_day` is given (see below), only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If :attr:`lead_day` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

    An instance of this class is configured via the :class:`traitlets.config.Application` interface, inherited from the :class:`.WRFiles` class in this module. All traitlet-based class/instance attributes (they appear in code as class attributes, but will only have configured values upon instantiation) can be configured via a config file, command-line arguments (see also `CEAZAMet stations webservice`_), or keyword arguments to the ``__init__`` call. (From the command-line, help can be obtained via the flag ``--help`` or ``--help-all``.)

    :Keyword arguments:
        Are the same as the :class:`traits <traitlets.TraitType>` described for this class.

    The following :class:`traits <traitlets.TraitType>` are defined on the :class:`.WRFiles` class:
        * :attr:`~.WRFiles.paths`
        * :attr:`~.WRFiles.domain`
        * :attr:`~.WRFiles.hour`
        * :attr:`~.WRFiles.from_date`
        * :attr:`~.WRFiles.directory_pattern`
        * :attr:`~.WRFiles.wrfout_prefix`
        * :attr:`data.CEAZA.Meta.file_name`

    """
    max_workers = Integer(16, help="number of threads to be used").tag(config=True)
    "Maximum number of threads to use."

    interpolator = Unicode('scipy').tag(config=True)
    """Which interpolator (if any) to use: ``scipy`` - use :class:`~data.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~data.interpolate.BilinearInterpolator`."""

    outfile = Unicode('out.nc').tag(config=True)
    """Base name of the output netCDF file (**no extension**, numberings are added in case of restarts, in the form '_part#' where # is the number). Defaults to the :attr:`outfile` trait which can be set by command line ('-o' flag) and config."""

    write_interval = Integer(73).tag(config=True) # 365 / 73 = 5
    """The number of input directories to process before triggering a write-out to file (only applicable if using :meth:`start` or :meth:`concat` with argument ``interval=True``)."""

    variables = Unicode().tag(config=True)
    """Name of variable(s) to extract."""

    interpolate = Bool(False).tag(config=True)
    """Whether or not to interpolate to station locations (see :class:`.Concatenator`)."""

    utc_delta = Instance(timedelta, kw={'hours': -4})
    """The offset from UTC to be applied to the concatenated data (assuming the original files are in UTC)."""

    lead_day = Integer(-1).tag(config=True)
    """Lead day of the forecast for which to search, if only one particular lead day is desired. (``-1`` denotes no particular lead day.)"""

    function = Unicode().tag(config=True)
    """Callable to be applied to the data before concatenation (after interpolation), in dotted from ('<module>.<function>')."""

    aliases = {'d': 'Concatenator.domain',
               'o': 'Concatenator.outfile',
               'v': 'Concatenator.variables'}

    flags = {'i': ({'Concatenator': {'interpolate': True}}, "interpolate to station locations")}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._var_list = [self.variables]
        self.data = None

    def func(self, value):
        try:
            return self._func(value)
        except AttributeError:
            if self.function == '':
                self._func = lambda x: x
            else:
                mod, f = os.path.splitext(self.function)
                self._func = getattr(import_module(mod), f)
            return self._func(value)

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
                start = timer()
                self.concat(True)
                self.write('final', start)
            except:
                self.write('inter', start)
                break

    def sort_data(self):
        if self.lead_day == -1:
            self.data = self.data.sortby('start')
        else:
            self.data = self.data.rename({'Time': 't'}).stack(Time=('start', 't')).sortby('XTIME')

    def write(self, middle, start):
        fn = '{}_{}_{}.nc'.format(self.outfile, middle, self.fileno)
        self.sort_data()
        self.data.to_netcdf(fn)
        self.remove_dirs(self.data)
        with open('timing.txt', 'a') as f:
            f.write('{} dirs in {} seconds, file {}\n'.format(
                len(self.data.start), timer() - start, fn))
        self.data = None
        self.fileno += 1

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - self.utc_delta # subtraction because we're going in the reverse direction here
        s = [d.strftime('%Y%m%d%H') for d in t]  # (from modified time in previous_file to original WRFOUT)
        n = len(self.dirs)
        self.dirs = [d for d in self.dirs if d[-10:] not in s]
        self.log.info("%s directories removed.", n - len(self.dirs))

    def concat(self, interval=False):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are found from the HDF file specified as :attr:`CEAZA.Meta.file_name`.

        :param interval: Whether or not to write out files at the interval specified by :attr:`write_interval`. This is mostly intended for internal use when :meth:`concat` is called from :meth:`start`

        """
        if self.interpolate:
            if not hasattr(self, '_interpolator'):
                f = glob(os.path.join(self.dirs[0], self.file_glob))
                with xr.open_dataset(f[0]) as ds:
                    self._interpolator = getattr(import_module('data.interpolate'), {
                        'scipy': 'GridInterpolator',
                        'bilinear': 'BilinearInterpolator'
                    }[self.interpolator])(ds, stations = self.stations)
            self.log.info("Interpolation requested with interpolator %s", self.interpolator)

        start = timer()
        while self.data is None:
            self.data = self._extract(self.dirs[0])
        if len(self.dirs) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.dirs)-1)) as exe:
                for i, f in enumerate(exe.map(self._extract, self.dirs[1:])):
                    self.data = xr.concat((self.data, f), 'start')
                    if interval and ((i + 1) % self.write_interval == 0):
                        self.write('reg', i+1, start)
            self.sort_data()
        print('Time taken: {:.2f}'.format(timer() - start))

    def _extract(self, d):
        try:
            with xr.open_mfdataset(os.path.join(d, self.file_glob)) as ds:
                print('using: {}'.format(ds.START_DATE))
                x = ds[self._var_list].sortby('XTIME') # this seems to be at the root of Dask warnings
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

if __name__ == '__main__':
    app = Concatenator(variables='T2', outfile='/HPC/arno/data/T2_new')
    # app = Concatenator()
    app.parse_command_line() # initialize() meant to be overridden
    app.start()
