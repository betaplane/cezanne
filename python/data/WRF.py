#!/usr/bin/env python
"""
WRFOUT concatenation
--------------------

Example Usage::

    from data import WRF
    w = WRF.Concatenator(domain='d03', interpolator='bilinear')
    w.run('T2-PSFC', var=['T2', 'PSFC'], interpolate=True)

.. NOTE::

    * ProcessPoolExecutor can't be used if unpicklable objects need to be exchanged.
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
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from netCDF4 import Dataset, num2date
from functools import partial
from timeit import default_timer as timer
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
            for d in glob(pa.join(p, pattern)):
                if (pa.isdir(d) and not pa.islink(d)):
                    dirs.append(d)
                    if limit is not None and len(dirs) == limit:
                        break
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]
        self.dirs = sorted(dirs)

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


    def __init__(self, domain, paths=None, hour=None, from_date=None, stations=None, interpolator='scipy', prefix='wrfout', dt=-4):
        self.dt = np.timedelta64(dt, 'h')
        glob_pattern = '{}_{}_*'.format(prefix, domain)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_proc = self.comm.Get_size()
        if self.rank == 0:
            F = Files(paths, hour, from_date)
            self.files = [f for d in F.dirs for f in glob(join(d, glob_pattern))]
            assert len(self.files) > 0, "no files added"

        if stations is not None:
            self._stations = stations

        if interpolator is not None:
            with Dataset(self.files[0]) as ds:
                self.intp = getattr(import_module('data.interpolate'),
                                    {'bilinear': 'BilinearInterpolator',
                                     'scipy': 'GridInterpolator'}[interpolator]
                                    )(ds, self.stations)

        print('WRF.Concatenator initialized with {} files'.format(len(self.files)))


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


    def concat(self, var, interpolate=None, lead_day=None, func=None):
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
        start = MPI.Wtime()
        variables = np.array([var]).flatten()

        with Dataset(self.files[0]) as ds:
            self.create_outfile(ds, variables, lead_day, name='out.nc')

        if self.rank > 0:
            while True:
                f = self.comm.recv(source=0, tag=1)
                print(f, self.rank)
                if f is None:
                    break
                self._extract(variables, lead_day, interpolate, func, f)
        else:
            while len(self.files) > 0:
                f = self.files.pop(0)
                for i in range(1, self.n_proc):
                    if len(self.files) == 0:
                        self.comm.send(None, dest=i, tag=1)
                        break
                    self.comm.send(self.files.pop(0), dest=i, tag=1)
                self._extract(variables, lead_day, interpolate, func, f)

        self.out.close()
        print('Time taken: {} (proc {})'.format(MPI.Wtime() - start), self.rank)

    def _extract(self, variables, lead_day, interp, func, f):
        print('file: {} (proc {})'.format(f, self.rank))
        ds = Dataset(f)

        xt = np.array((lambda t: num2date(t[:], t.units))(ds['XTIME']), dtype='datetime64') + self.dt
        for i, v in enumerate(variables):
            var = ds[v]
            if lead_day is None:
                x = np.expand_dims(var[:], 0)
                dims = np.r_[['start'], var.dimensions]
            else:
                t = pd.DatetimeIndex(xt)
                idx = (t - t.min()).days == lead_day
                x = var[[idx if d=='Time' else slice(None) for d in var.dimensions]]
                dims = var.dimensions
            print('recv', (self.rank-1)%self.n_proc, 100+i)
            start = self.comm.recv(source=(self.rank-1) % self.n_proc, tag=100+i)
            start_slice = slice(start, start+1)
            self.out[v][[{
                'start': start_slice,
                'Time': slice(None, ds.dimensions['Time'].size)
            }.get(d, slice(None)) for d in dims]] = x

            self.comm.send(self.out[v].shape[0], dest=(self.rank+1) % self.n_proc, tag=100+i)
        if lead_day is None:
            self.out['start'][start_slice] = xt.min()

    def create_outfile(self, ds, variables, lead_day, name='out.nc'):
        self.out = Dataset(name, 'w', parallel=True, format="NETCDF4")
        coords = {'XTIME'}
        dims = {'Time'}
        self.out.createDimension('Time', None)
        if lead_day is None:
            self.out.createDimension('start', None)
            self.out.createVariable('start', ds['XTIME'].dtype, ('start'))
            self.out['start'].set_collective(True)
            self.out.createVariable('XTIME', ds['XTIME'].dtype, ('start', 'Time'))
        else:
            self.out.createVariable('XTIME', ds['XTIME'].dtype, ('Time'))
        self.out['XTME'].set_collective(True)
        for j, v in enumerate(variables):
            var = ds[v]
            for i, d in enumerate(set(var.dimensions) - dims):
                dim = ds.dimensions[d]
                self.out.createDimension(d, dim.size)
            # as is this works probably only for horizontal coordinates (XLONG, XLAT, staggered maybe)
            for c in set(var.coordinates.split()) - coords:
                x = ds[c]
                self.out.createVariable(x.name, x.dtype, x.dimensions[1:])
                self.out[x.name][:] = x[1, :, :]
            if lead_day is None:
                self.out.createVariable(v, var.dtype, np.r_[['start'], var.dimensions])
            else:
                self.out.createVariable(v, var.dtype, var.dimensions)

            self.out[v].set_collective(True)
            dims = dims.union(var.dimensions)
            coords = coords.union(var.coordinates.split())


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
