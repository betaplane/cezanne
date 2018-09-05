#!/usr/bin/env python
"""
WRF output processing
---------------------

There are two submodules with roughly identical APIs in this package, one using the :class:`concurrent.futures.ThreadPoolExecutor` interface, and one using :mod:`mpi4pi.MPI`. This file (/base.py/) contains shared code, in particular the classes :class:`.WuRFiles`, which centralizes dealing with WRF output files distributed over various storage elements and can be used to obtain these files directly, and :class:`.CCBase`, which is only the container for code share between the concatenator subclasses :class:`.threadWuRF.CC` and :class:`.mpiWuRF.CC`.

Common interface options
========================

All options to the interfaces are specified using the `traitlets.config  <https://traitlets.readthedocs.io/en/stable/config.html>`_ system. The appear in code as class attributes (:class:`traits <traitlets.TraitType>`) which will be populated with values once the class is instantiated. They can be specified as arguments to the class instantiation call (as keyword arguments with the same name as the traits), in a :attr:`config file <WuRFiles.config_file>`, and as `command line arguments <https://traitlets.readthedocs.io/en/stable/config.html#command-line-arguments>`_ when running a file as script.

The interface options common to both concatenators are in part those of the :class:`.WurFiles`:
    * :attr:`~.WuRFiles.paths`
    * :attr:`~.WuRFiles.domain`
    * :attr:`~.WuRFiles.hour`
    * :attr:`~.WuRFiles.from_date`
    * :attr:`~.WuRFiles.directory_pattern`
    * :attr:`~.WuRFiles.wrfout_prefix`
    * :attr:`~.WuRFiles.max_workers`

Further common options are gathered in the :class:`.CCBase` class:
    * :attr:`~.CCBase.outfile`
    * :attr:`~.CCBase.variables`
    * :attr:`~.CCBase.interpolator`
    * :attr:`~.CCBase.interpolate`
    * :attr:`~.CCBase.utc_delta`
    * :attr:`~.CCBase.lead_day`
    * :attr:`~.CCBase.function`

The :class:`~pandas.DataFrame` with station locations (for :mod:`interpolation <data.interpolate>`) is currently specified as :attr:`data.CEAZA.Meta.file_name` of the :mod:`~data.CEAZA` module.

Aliases for the command line options are defined by the various :class:`traitlets.config.application.Application` interfaces (:class:`.threadWuRF.CC`, :class:`.tests.WuRFTests`) and can be `queried for help  <https://traitlets.readthedocs.io/en/stable/config.html#subcommands>`_ by executing the respective file as a script with ``--help`` or ``--help-all`` as arguments.

.. _WuRF-usage:

Usage
=====

The following invocations are equivalent. From a python script / REPL::

    import WuRF
    w = WuRF.Concatenator(variables='T2', outfile='T2', domain='d03', interpolator='bilinear', interpolate=True)
    w.start()

From the command line::

    WuRF/threads.py -v T2 -o T2 -d 'd03' --Concatenator.interpolator='bilinear' -i

If :attr:`~.CCBase.interpolate` is ``True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config`.

If :attr:`~.CCBase.lead_day` is given (see below), only the day's data with the given lead is taken from each daily simulation, resultng in a continuous temporal sequence. If it is not given (i.e. has a value of ``-1``), the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

There is one difference between the two modules: :class:`.threadWuRF.CC` has two methods, :meth:`~.threadWuRF.CC.concat`, and :meth:`.threadWuRF.CC.start`, of which the former does not write out a file unless being called with the argument ``interval=True``, while the latter is a wrapper specifically to write out the data to file (periodically, with the interval specifiec by :attr:`~.threadWuRF.CC.write_interval` and in case of error). The MPI-based module continuously writes out a netCDF4_ in parallel ans has only a :meth:`.mpiWuRF.CC.start` method.

.. NOTE::

    * The :class:`.mpiWuRF.CC` doesn't implement the :attr:`.function` interface yet.
    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).
    * Data for the tests can be found in the directory specified by :attr:`tests.WuRFTests.test_dir`.
    * The :class:`Concatenator` class uses the following attributes and methods from its super:
        * dirs
        * stations / _stations
        * file_glob / _file_glob
    * xr.concat works if Time dim has different lengths **if it has a coordinate** (right now it doesn't)
    * **opening files as xr.open_dataarray throws away the metadata of the files**

.. WARNING::

    The current layout with ThreadPoolExecutor seems to work on the UV only if one sets::

        export OMP_NUM_THREADS=1

    (`a conflict between OpenMP and dask? <https://stackoverflow.com/questions/39422092/error-with-omp-num-threads-when-using-dask-distributed>`_)

.. TODO::

    * add coordinate to Time?
    * custom logger with automatic parallel info (rank etc)
    * implement func application (MPI)
    * maybe data that needs to be shared can be loaded onto the class **before** initializing MPI????


.. automodule:: python.WuRF.threadWuRF
    :members:

.. automodule:: python.WuRF.mpiWuRF
    :members:

.. automodule:: python.WuRF.tests
    :members:

"""
__package__ = 'WuRF'
import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime, timedelta
from traitlets.config import Application, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import List, Integer, Unicode, Bool, Instance
from functools import partial
from importlib import import_module


def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.CEAZAMet`) with a concatenated (and interpolated to station locations) netCDF file as produced by this packages concatenators (:class:`threadWuRF.CC`, :class:`mpiWuRF.CC`). For now, works with a single field.

    :param wrf: The DataArray or Dataset containing the concatenated and interpolated (to station locations) WRF simulations (only dimensions ``start`` and ``Time`` and coordinate ``XTIME`` are used).
    :type wrf: :class:`~xarray.DataArray` or :class:`~xarray.Dataset`
    :param df: The DataFrame containing the station data (of the shape returned by :meth:`.CEAZA.Downloader.get_field`).
    :type df: :class:`~pandas.DataFrame`
    :returns: DataArray with ``start`` and ``Time`` dimensions aligned with **wrf**.
    :rtype: :class:`~xarray.DataArray`

    """
    xr = import_module('xarray')
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    # necessary because some timestamps seem to be slightly off-round hours
    xt = xr.DataArray(pd.Series(xt.values).dt.round('h'), coords=xt.coords).unstack('t')
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    cols = df.columns.get_level_values('station').intersection(wrf.station)
    return xr.DataArray(np.stack([np.where(idx>=0, df[c].values[idx].squeeze(), np.nan) for c in cols], 2),
                     coords = [wrf.coords['start'], wrf.coords['Time'], ('station', cols)])

class WuRFiles(Application):
    """Base class for the WRF-file concatenators in this package (:mod:`.threads` and :mod:`.MPI`). It retrieves the original WRF output files and is configured/run via the :class:`traitlets.config.Application` interface.

    This class can be used by itself to retrieve WRF output files in exactly the same manner as the concatenators do.

    """
    max_workers = Integer(16, help="number of threads to be used").tag(config=True)
    "Maximum number of threads to use."

    paths = List(help="list of paths where WRFOUT files can be found").tag(config = True)
    """List of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs."""

    directory_pattern = Unicode('c01_*', help="").tag(config = True)
    """Glob pattern for the directory names (e.g. ``c01_*``)"""

    wrfout_prefix = Unicode('wrfout').tag(config = True)
    """Prefix of the WRF output files (e.g. ``wrfout``)."""

    hour = Integer(-1).tag(config = True)
    """Hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired."""

    from_date = Unicode().tag(config = True)
    """From what date onwards to search (only simulation start dates), if a start date is desired (as %Y%m%d :obj:`str`)."""

    domain = Unicode('d03').tag(config=True)
    """Which of the WRF domains (e.g. ``d03``) to use."""

    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    """path to config file"""

    aliases = {'m': 'Meta.file_name'}

    def __init__(self, limit=None, **kwargs):
        super().__init__(**kwargs)
        try:
            config = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            config.merge(self.config)
            self.update_config(config)
        except ConfigFileNotFound:
            pass

        self.paths = [p for p in self.paths if os.path.isdir(p)]

        dirs = []
        for p in self.paths:
            for d in sorted(glob(os.path.join(p, self.directory_pattern))):
                if (os.path.isdir(d) and not os.path.islink(d)):
                    dirs.append(d)
                    if limit is not None and len(dirs) == limit:
                        break
        dirs = dirs if self.hour == -1 else [d for d in dirs if d[-2:] == '{:02}'.format(self.hour)]
        self.dirs = dirs if self.from_date == '' else [d for d in dirs if d[-10:-2] >= self.from_date]

        assert len(self.dirs) > 0, "no directories added"
        self.log.info("WuRFiles initialized with %s directories.", len(self.dirs))

    @staticmethod
    def _name(domain, lead_day, prefix, d):
        ts = pd.Timestamp.strptime(os.path.split(d)[1][-10:], '%Y%m%d%H')
        if lead_day is None:
            g = glob(os.path.join(d, '{}_{}_*'.format(prefix, domain)))
        else:
            s = (ts + pd.Timedelta(lead_day, 'D')).strftime('%Y-%m-%d_%H:%M:%S')
            g = [f for f in glob(os.path.join(d, '{}_{}_{}'.format(prefix, domain, s))) if os.path.isfile(f)]
        return (g, len(g), ts.hour)

    def files(self, domain=None, lead_day=None, prefix='wrfout'):
        name = partial(self._name, domain, lead_day, prefix)
        self.files, self.length, self.hour = zip(*[name(d) for d in self.dirs])

    def by_sim_length(self, n):
        return [f for d in np.array(self.files)[np.array(self.length) == n] for f in d]

    @classmethod
    def first(cls, domain, lead_day=None, hour=None, from_date=None, pattern=None, prefix='wrfout', opened=True):
        """Get the first netCDF file matching the given arguments (see :class:`CC` for a description), based on the configuration values (section *wrfout*) in the global config file.

        """
        f = cls(hour=hour, from_date=from_date, pattern=pattern, limit=1)
        name = partial(cls._name, domain, lead_day, prefix)
        files, _, _ = name(f.dirs[0])
        if opened:
            xr = import_module('xarray')
            return xr.open_dataset(files[0])
        return files[0]

    @property
    def stations(self):
        try:
            return self._stations
        except AttributeError:
            self._stations = pd.read_hdf(self.config.Meta.file_name, 'stations')
            return self._stations

    @property
    def file_glob(self):
        try:
            return self._file_glob
        except AttributeError:
            self._file_glob = '{}_{}_*'.format(self.wrfout_prefix, self.domain)
            return self._file_glob

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - self.utc_delta # subtraction because we're going in the reverse direction here
        s = [d.strftime('%Y%m%d%H') for d in t]  # (from modified time in previous_file to original WRFOUT)
        n = len(self.dirs)
        self.dirs = [d for d in self.dirs if d[-10:] not in s]
        self.log.info("%s directories removed.", n - len(self.dirs))

    def duration(self):
        """Save a list of simulation lengths (in timesteps), per directory in :attr:`.dirs`. The list is saved in :attr:`.lengths`."""
        # saved in /HPC/arno/data/wrf_dir_stats.h5 so it can be simply updated (runs a long time)
        # current sim length is 145
        xr = import_module('xarray')
        tq = import_module('tqdm')
        with tq.tqdm(total = len(self.dirs)) as prog:
            self.lengths = []
            for d in self.dirs:
                try:
                    self.lengths.append((d, len(xr.open_mfdataset(os.path.join(d, self.file_glob)).Time)))
                    prog.update(1)
                except OSError as err:
                    print('{}: {}'.format(err, d))

    def dates_from_dirs(self):
        s = pd.Index(self.dirs).str.extract('_(\d+)$').to_series()
        s = s.apply(lambda s: datetime.strptime(s, '%Y%m%d%H'))
        s.index = pd.Index([os.path.split(d) for d in self.dirs])
        return s

    @staticmethod
    def duration_blocks(df):
        return [(l, (lambda i:(i.min(), i.max()))(df.index.str.extract('_(\d+)$')[df.length==l]))
         for l in df.length.unique()]

    def version(self):
        """Extract 'TITLE' string (containing version info) from all directories"""
        xr = import_module('xarray')
        tq = import_module('tqdm')
        with tq.tqdm(total = len(self.dirs)) as prog:
            self.versions = []
            for d in self.dirs:
                try:
                    self.versions.append((d, xr.open_mfdataset(os.path.join(d, self.file_glob)).TITLE))
                    prog.update(1)
                except OSError as err:
                    print('{}: {}'.format(err, d))

    def meta_changes(self):
        xr = import_module('xarray')
        tq = import_module('tqdm')
        with tq.tqdm(total = len(self.dirs)) as prog:
            self.attrs = []
            for d in self.dirs:
                try:
                    ds = xr.open_dataset(glob(os.path.join(d, '*d03*'))[0])
                    a = ds.attrs.copy()
                    for k in ['START_DATE', 'SIMULATION_START_DATE', 'JULDAY']:
                        a.pop(k)
                    try:
                        b = [(k, v) for k, v in a.items() if prev[k] != v]
                        if len(b) > 0:
                            self.attrs.append((d, b))
                    except Exception as err:
                        self.attrs.append((d, err))
                    prev = a
                except Exception as err:
                    self.attrs.append((d, err))
                prog.update(1)

class CCBase(WuRFiles):

    outfile = Unicode('out.nc').tag(config=True)
    """Base name of the output netCDF file (**no extension**, numberings are added in case of restarts, in the form '_part#' where # is the number). Defaults to the :attr:`outfile` trait which can be set by command line ('-o' flag) and config."""

    interpolator = Unicode('scipy').tag(config=True)
    """Which interpolator (if any) to use: ``scipy`` - use :class:`~data.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~data.interpolate.BilinearInterpolator`."""

    variables = Unicode().tag(config=True)
    """Name of variable(s) to extract."""

    interpolate = Bool(False).tag(config=True)
    """Whether or not to interpolate to station locations (see :ref:`WuRF-usage`)."""

    utc_delta = Instance(timedelta, kw={'hours': -4})
    """The offset from UTC to be applied to the concatenated data (assuming the original files are in UTC)."""

    lead_day = Integer(-1).tag(config=True)
    """Lead day of the forecast for which to search, if only one particular lead day is desired. (``-1`` denotes no particular lead day.)"""

    function = Unicode().tag(config=True)
    """Callable to be applied to the data before concatenation (after interpolation), in dotted from ('<module>.<function>'). (**Not implemented in :mod:`.mpiWuRF` yet**)"""

    aliases = {'d': 'WuRFiles.domain',
               'o': 'CCBase.outfile',
               'v': 'CCBase.variables'}

    flags = {'i': ({'CCBase': {'interpolate': True}}, "interpolate to station locations")}

    @property
    def var_list(self):
        return [s.strip() for s in self.variables.split(',')]

    def func(self, value):
        try:
            return self._func(value)
        except AttributeError:
            if self.function == '':
                self._func = lambda x: x
            else:
                mod, f = os.path.splitext(self.function)
                self._func = getattr(import_module(mod), f.strip('.'))
            return self._func(value)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

def piece_together(glob_pattern):
    """Piece together individual files resulting from a WRF output concatenation procedure (due to different simulation lengths and random interruptions) into larger files.

    """
    xr = import_module('xarray')
    D = {}
    t = {133: 121, 169: 145}
    for g in sorted(glob(glob_pattern)):
        f = xr.open_dataset(g)
        j = len(f.Time)
        d = f.sel(Time=slice(0, t[j]))
        i = len(d.Time)
        try:
            D[i] = xr.concat((D[i], d), 'start')
        except KeyError:
            D[i] = d
    for f in D.items():
        f.to_netcdf('{}_{}.nc'.format(glob_pattern[:-1], l))


def tease_apart(glob_pattern, var):
    """Separate a number of files containing concatenated WRF output and given by a glob pattern according to known breaks in our operational WRF simulations.

    Currently known breaks are:

        * Change from 0:00 UTC simulation start time to 12:00 UTC (April 2016). This change is nearly concurrent with a version upgrade of WRF from 3.7 to 3.8
        * Change in GFS in July 2017, which requires a new version of WPS to be used (3.9.0.1) - this was initially not implemented immediately, and hence the simulations with configuration "c01" after this date should be discarded. Configuration "c05" should not be affected and is used in newer operational runs.

    :returns: Tuple of :class:`xarray.Dataset` separated at the relevant time points.
    :rtype: :obj:`tuple`

    """
    xr = import_module('xarray')
    ds = [xr.open_dataset(f)[var] for f in glob(glob_pattern)]
    n = min(len(d.Time) for d in ds)
    # using .isel() is essential b/c .sel() *includes* the end of slice
    x = xr.concat([d.isel(Time=slice(None, n)) for d in ds], 'start').sortby('start')
    h = x.start.to_index().hour
    return x.sel(start = h==20), x.sel(start = h==8).sel(start=slice(None, '2017-07-18'))
