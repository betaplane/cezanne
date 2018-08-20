#!/usr/bin/env python
"""
WRFOUT concatenation (MPI version)
----------------------------------

Example Usage::

    from data import WRF
    w = WRF.Concatenator(domain='d03', interpolator='bilinear')
    w.concat(variables=['T2', 'PSFC'], out_name='T2-PSFC' , interpolate=True)

.. NOTE::

    * The wrfout files contain a whole multi-day simulation in one file starting on March 7, 2018 (instead of one day per file as before).

"""
from glob import glob, os
import pandas as pd
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset, MFDataset, num2date, date2num
from datetime import datetime, timedelta
from functools import partial
from importlib import import_module
from traitlets.config.loader import PyFileConfigLoader

config = PyFileConfigLoader(os.path.expanduser('~/Dropbox/work/config.py')).load_config()


class Concatenator(object):
    """WRFOUT file concatenator (MPI version), for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).

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
            self.dirs = F.dirs
            assert len(F.dirs) > 0, "no files added"
            # so that the first file is indeed the first, for consistency with the date2num units
            self._first = min(glob(pa.join(self.dirs[0], self._glob_pattern)))
        else:
            self._first = None
        self._first = self.comm.bcast(self._first, root=0)

        if interpolator is not None:
            lonlat = self.stations(stations)
            with Dataset(self._first) as ds:
                self.intp = getattr(import_module('data.interpolate'),
                                    {'bilinear': 'BilinearInterpolator',
                                     'scipy': 'GridInterpolator'}[interpolator]
                                    )(ds, **lonlat)

        if self.rank == 0:
            print('Concatenator initialized with {} dirs, interpolator {}'.format(len(F.dirs), interpolator))

    def stations(self, sta):
        # HDFStore can only be opened by one process
        if self.rank == 0:
            if sta is None:
                with pd.HDFStore(config['stations']['sta']) as sta:
                    sta = sta['stations']
            lon, lat = sta[['lon', 'lat']].as_matrix().T
            self.n_stations = lon.size
            self._names = sta.index.values
            ll = {'lon': lon, 'lat': lat}
        else:
            ll = None
        ll = self.comm.bcast(ll, root=0)
        return ll

    def remove_dirs(self, ds):
        t = ds.indexes['start'] - pd.Timedelta(self.dt, 'h')
        s = [d.strftime('%Y%m%d%H') for d in t]
        return [d for d in self.dirs if d[-10:] not in s]

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

        self._create_outfile(out_name, var, lead_day, interpolate)

        self.start = 0
        n_dirs = len(self.dirs) if self.rank == 0 else None
        n_dirs = self.comm.bcast(n_dirs, root=0)
        while n_dirs >= self.n_proc:
            if self.rank == 0:
                dirs = self.dirs[:self.n_proc]
                self.dirs = self.dirs[self.n_proc:]
            else:
                dirs = None
            # this only works of scatter maintains the order
            # so far it has always done so...
            dirs = self.comm.scatter(dirs, root=0)
            n_dirs = n_dirs - self.n_proc

            self._extract(out_name, dirs, var, lead_day, interpolate, parallel=True)

        if self.rank == 0:
            for i in range(n_dirs):
                self._extract(out_name, self.dirs[i], var, lead_day, interpolate, parallel=False)

        print('Time taken: {} (proc {})'.format(MPI.Wtime() - start, self.rank))

    def _dimsize(self, ds, d):
        if d=='start':
            return self.start + self.rank
        if d=='station':
            return slice(None)
        dim = ds.dimensions[d]
        if hasattr(dim, 'size'):
            return slice(None, dim.size)
        elif hasattr(dim, 'dimlens'):
            return slice(None, sum(dim.dimlens))

    def _extract(self, out_name, d, variables, lead_day, interp, parallel=True):
        print('dir: {} (proc {})'.format(d, self.rank))

        # somehow, it didn't seem possible to have a file open in parallel mode and perform the
        # scatter operation
        out = Dataset(out_name, 'a', format='NETCDF4', parallel=parallel)
        if parallel:
            out['XTIME'].set_collective(True)
            if lead_day is None:
                out['start'].set_collective(True)
            for v in variables:
                out[v].set_collective(True)

        with MFDataset(pa.join(d, self._glob_pattern)) as ds:
            xtime = ds['XTIME']
            xt = np.array(num2date(xtime[:], xtime.units), dtype='datetime64[m]') + self.dt
            t = date2num(xt.astype(datetime), units=self.time_units)
            if lead_day is None:
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
                x = var[:]
                dims = var.dimensions
                if interp:
                    x, dims = self.intp.netcdf(var)
                if lead_day is None:
                    x = np.expand_dims(x, 0)
                    D = [self._dimsize(ds, d) for d in np.r_[['start'], dims]]
                else:
                    x = x[[idx if d=='Time' else slice(None) for d in dims]]
                    D = [t_slice if d=='Time' else slice(None) for d in dims]

                out[v][D] = x

        out.close()
        self.start = self.start + self.n_proc if parallel else self.start + 1

    # all very WRF-specific
    # only called by rank 0
    def _dims_and_coords(self, ds, var, interp):
        coords = set.union(*[set(ds[v].coordinates.split()) for v in var]) - {'XTIME'}
        dims = set.union(*[set(ds[v].dimensions) for v in var]) - {'Time'}

        D, C, V = [], [], []
        for d in dims:
            if (not interp) or (d not in self.intp.spatial_dims):
                D.append((d, ds.dimensions[d].size))

        for c in coords:
            d = ds[c].dimensions[1:]
            if (not interp) or (len(set(d).intersection(self.intp.spatial_dims)) == 0):
                C.append((c, ds[c].dtype, d))

        for v in var:
            dims = [d for d in ds[v].dimensions if ((not interp) or (d not in self.intp.spatial_dims))]
            if interp:
                dims.insert(0, 'station')
            V.append((v, ds[v].dtype, dims))

        if interp:
            D.insert(0, ('station', self.n_stations))
            C.insert(0, ('station', str, 'station'))

        return D, C, V

    # file has to be created and opened BY ALL PROCESSES
    def _create_outfile(self, name, var, lead_day, interp):
        # I think there's a file formate issue - I can open the WRF files only in single-proc mode
        if self.rank == 0:
            ds = Dataset(self._first)
            dcv = self._dims_and_coords(ds, var, interp)
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

        # the reason for using OrderedDicts in _dims_and_coords is that otherwise the order
        # of the items is not guaranteed to be the same for all procs, and thus the loops
        # here would deadlock since the operations are collective
        for i in dims:
            out.createDimension(*i)

        for c in coords:
            out.createVariable(*c)

        for v in variables:
            out.createVariable(*v[:2], np.r_[['start'], v[2]] if lead_day is None else v[2])

        # apparently, writing in non-parallel mode is fine
        # but I think it needs to be done after defining, as in F90/C
        if self.rank == 0:
            for k in coords:
                if k[0] != 'station':
                    out[k[0]][:] = ds[k[0]][1, :, :] # WRF-specific (coords for each timestep)
            ds.close()
        out.close()

        # however, string variables are treated as 'vlen' and having an unlimited dimension, which
        # would require collective parallel writing
        if interp and self.rank == 0:
            with Dataset(name, 'a') as out:
                out['station'][:] = self._names
