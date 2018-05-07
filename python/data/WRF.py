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
    * To run tests with mpi4py (the import form is necessary because of the config values only accessibly via the parent package :mod:`data`)::

        mpiexec -n 1 python -c "from data import tests; tests.run_tests()"

    * If using :mod:`condor`, the code files need to be downloaded for the tests to work correctly (because of relative imports in the :mod:`data` package)::

        mpiexec -n 1 python -c "import condor; condor.enable_sshfs_import(..., download=True); from data import tests; tests.run_tests()"
.. TODO::


"""
from glob import glob
import os.path as pa
import pandas as pd
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset, MFDataset, num2date, date2num
from datetime import datetime, timedelta
from functools import partial
from importlib import import_module
import unittest
from . import config


def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.Downloader`) with a concatenated (and interpolated to station locations) netCDF file as produced by :class:`.Concatenator`. For now, works with a single field.

    :param wrf: The DataArray or Dataset containing the concatenated and interpolated (to station locations) WRF simulations (only dimensions ``start`` and ``Time`` and coordinate ``XTIME`` are used).
    :type wrf: :class:`~xarray.DataArray` or :class:`~xarray.Dataset`
    :param df: The DataFrame containing3 the station data (of the shape returned by :meth:`.CEAZA.Downloader.get_field`).
    :type df: :class:`~pandas.DataFrame`
    :returns: DataArray with ``start`` and ``Time`` dimensions aligned with **wrf**.
    :rtype: :class:`~xarray.DataArray`

    """
    import xarray as xr
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    # necessary because some timestamps seem to be slightly off-round hours
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
            for d in glob(pa.join(p, pattern)):
                if (pa.isdir(d) and not pa.islink(d)):
                    dirs.append(d)
                    if limit is not None and len(dirs) == limit:
                        break
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]
        self.dirs = sorted(dirs, key=lambda d: d[-10:]) # <- sort by date

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
        """Get the first netCDF file matching the given arguments (see :class:`Concatenator` for a description), based on the configuration values (section *wrfout*) in the global config file. NOTE: the first matching file is taken before all the directories are sorted by date!

        """
        f = cls(hour=hour, from_date=from_date, pattern=pattern, limit=1)
        name = partial(cls._name, domain, lead_day, prefix)
        files, _, _ = name(f.dirs[0])
        return files[0]

    @staticmethod
    def stations():
        with pd.HDFStore(config['stations']['sta']) as S:
            return S['stations']


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
    time_units = 'minutes since 2015-01-01 00:00:00'
    time_type = np.float64

    def __init__(self, domain, paths=None, hour=None, from_date=None, stations=None, interpolator='scipy', prefix='wrfout', dt=-4):
        self.dt = np.timedelta64(dt, 'h')
        self._glob_pattern = '{}_{}_*'.format(prefix, domain)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_proc = self.comm.Get_size()
        if self.rank == 0:
            F = Files(paths, hour, from_date)
            self.files = F
            assert len(F.dirs) > 0, "no files added"
            # so that the first file is indeed the first, for consistency with the date2num units
            self._first = min(glob(pa.join(self.files.dirs[0], self._glob_pattern)))
        else:
            self._first = None
        self._first = self.comm.bcast(self._first, root=0)

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            with Dataset(self._first) as ds:
                self.intp = getattr(import_module('data.interpolate'),
                                    {'bilinear': 'BilinearInterpolator',
                                     'scipy': 'GridInterpolator'}[interpolator]
                                    )(ds, self.stations)

        if self.rank == 0:
            print('WRF.Concatenator initialized with {} dirs'.format(len(F.dirs)))

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


    def concat(self, variables, out_name='out.nc', interpolate=None, lead_day=None, func=None):
        """Concatenate the found WRFOUT files. If ``interpolate=True`` the data is interpolated to station locations; these are either given as argument instantiation of :class:`.Concatenator` or read in from the :class:`~pandas.HDFStore` specified in the :data:`.config`. If ``lead_day`` is given, only the day's data with the given lead is taken from each daily simulation, resulting in a continuous temporal sequence. If ``lead_day`` is not given, the data is arranged with two temporal dimensions: **start** and **Time**. **Start** refers to the start time of each daily simulation, whereas **Time** is simply an integer index of each simulation's time steps.

        :param variables: Name of variable to extract. Can be an iterable if several variables are to be extracted at the same time).
        :param out_name: Name of the output file to which to write the concatenated data.
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
        start = MPI.Wtime()
        var = np.array([variables]).flatten()

        self._create_outfile(out_name, var, lead_day)

        self.start = 0
        n_dirs = len(self.files.dirs) if self.rank == 0 else None
        n_dirs = self.comm.bcast(n_dirs, root=0)
        while n_dirs >= self.n_proc:
            if self.rank == 0:
                dirs = self.files.dirs[:self.n_proc]
                self.files.dirs = self.files.dirs[self.n_proc:]
            else:
                dirs = None
            dirs = self.comm.scatter(dirs, root=0)
            n_dirs = n_dirs - self.n_proc

            self._extract(out_name, dirs, var, lead_day, interpolate, func)

        while n_dirs > 0:
            if self.rank == 0:
                self._extract(out_name, dirs, var, lead_day, interpolate, func)

        print('Time taken: {} (proc {})'.format(MPI.Wtime() - start, self.rank))

    @staticmethod
    def _dimsize(dim):
        if hasattr(dim, 'size'):
            return slice(None, dim.size)
        elif hasattr(dim, 'dimlens'):
            return slice(None, sum(dim.dimlens))

    def _extract(self, out_name, d, variables, lead_day, interp, func):
        print('dir: {} (proc {})'.format(d, self.rank))

        # somehow, it didn't seem possible to have a file open in parallel mode and perform the
        # scatter operation
        out = Dataset(out_name, 'a', format='NETCDF4', parallel=True)

        with MFDataset(pa.join(d, self._glob_pattern)) as ds:
            xtime = ds['XTIME']
            xt = np.array(num2date(xtime[:], xtime.units), dtype='datetime64[m]') + self.dt
            t = date2num(xt.astype(datetime), units=self.time_units)
            out['XTIME'].set_collective(True)
            if lead_day is None:
                out['start'].set_collective(True)
                out['start'][self.start + self.rank] = t.min()
                out['XTIME'][self.start + self.rank, :xt.size] = t
            else:
                idx = (xt - xt.min()).astype('timedelta64[D]').astype(int) == lead_day
                n = idx.astype(int).sum()
                xout = out['XTIME']
                t_slice = slice(xout.size + self.rank * n, xout.size + (self.rank + 1) * n)
                xout[t_slice] = t[idx]

            for i, v in enumerate(variables):
                var = ds[v]
                if lead_day is None:
                    x = np.expand_dims(var[:], 0)
                    dims = np.r_[['start'], var.dimensions]
                    D = list(np.r_[[self.start + self.rank], [self._dimsize(ds.dimensions[d]) for d in var.dimensions]])
                else:
                    x = var[[idx if d=='Time' else slice(None) for d in var.dimensions]]
                    dims = var.dimensions
                    D = [t_slice if d=='Time' else slice(None) for d in var.dimensions]

                out[v].set_collective(True)
                out[v][D] = x

        out.close()
        self.start = self.start + self.n_proc

    @staticmethod
    def _dims_and_coords(ds, var):
        coords = set.union(*[set(ds[v].coordinates.split()) for v in var]) - {'XTIME'}
        dims = set.union(*[set(ds[v].dimensions) for v in var]) - {'Time'}

        dims = {d: ds.dimensions[d].size for d in dims}
        coords = {c: (ds[c].dtype, ds[c].dimensions[1:]) for c in coords}
        variables = {v: (ds[v].dtype, ds[v].dimensions) for v in var}
        return dims, coords, variables

    # file has to be created and opened BY ALL PROCESSES
    def _create_outfile(self, name, var, lead_day):
        # I think there's a file formate issue - I can open the WRF files only in single-proc mode
        if self.rank == 0:
            ds = Dataset(self._first)
            dcv = self._dims_and_coords(ds, var)
        else:
            dcv = None

        dcv = self.comm.bcast(dcv, root=0)
        dims, coords, variables = dcv

        out = Dataset(name, 'w', format='NETCDF4', parallel=True)

        out.createDimension('Time', None)
        if lead_day is None:
            out.createDimension('start', None)
            s = out.createVariable('start', self.time_type, ('start'))
            s.units = self.time_units
            x = out.createVariable('XTIME', self.time_type, ('start', 'Time'))
        else:
            x = out.createVariable('XTIME', self.time_type, ('Time'))
        x.units = self.time_units

        for i in dims.items():             # these loops need to be synchronized for parallel access!!!
            i = self.comm.bcast(i, root=0) # synchronization device - barrier() doesn't seem to work here
            out.createDimension(*i)

        for i in coords.items():
            i = self.comm.bcast(i, root=0)
            k, v = i
            c = out.createVariable(k, v[0], v[1])

        for i in variables.items():
            i = self.comm.bcast(i, root=0)
            k, v = i
            out.createVariable(k, v[0], np.r_[['start'], v[1]] if lead_day is None else v[1])

        # apparently, writing in non-parallel mode is fine
        # but I think it needs to be done after defining, as in F90/C
        if self.rank == 0:
            for k in coords.keys():
                out[k][:] = ds[k][1, :, :]
            ds.close()
        out.close()




if __name__ == '__main__':
    unittest.main()
