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
    * I don't think application of a function is currently supported.

.. TODO::
    * custom logger with automatic parallel info (rank etc)
    * implement func application
    * maybe data that needs to be shared can be loaded onto the class **before** initializing MPI????

"""
from mpi4py import MPI
from netCDF4 import Dataset, MFDataset, num2date, date2num
from datetime import datetime, timedelta
from .concat import *

class Concatenator(CCBase):
    """WRFOUT file concatenator (MPI version), for a specifc forecast lead day or for all data arrange in two temporal dimensions, and with (optional) interpolation to station location (see :meth:`.concat` for details).

    :param domain: Domain specifier for which to search among WRFOUT-files.
    :param paths: list of names of base directories containing the 'c01\_...' directories corresponding to individual forecast runs
    :param hour: hour at which the forecast starts (because we switched from 0h to 12h UTC), if selection by hour is desired.
    :param from_date: From what date onwards to search (only simulation start dates), if a start date is desired.
    :type from_date: %Y%m%d :obj:`str`
    :param stations: DataFrame with station locations (as returned by a call to :meth:`.CEAZA.Downloader.get_stations`) to which the variable(s) is (are) to be interpolated, or expression which evaluates to ``True`` if the default station data is to be used (as saved in :data:`config_file`).
    :param interpolator: Which interpolator (if any) to use: ``scipy`` - use :class:`~.interpolate.GridInterpolator`; ``bilinear`` - use :class:`~.interpolate.BilinearInterpolator`.

    """
    time_units = Unicode('minutes since 2015-01-01 00:00:00').tag(config=True)
    time_type = np.float64

    def start(self):
        # so that the first file is indeed the first, for consistency with the date2num units
        self._first = min(glob(os.path.join(self.dirs[0], self.file_glob)))

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_proc = self.comm.Get_size()
        start = MPI.Wtime()

        if self.interpolate:
            if not hasattr(self, '_interpolator'):
                with Dataset(self._first) as ds:
                    self._interpolator = getattr(import_module('data.interpolate'), {
                        'bilinear': 'BilinearInterpolator',
                        'scipy': 'GridInterpolator'}[self.interpolator])(ds, **self.lonlat)
            self.log.info("Interpolation requested with interpolator %s", self.interpolator)

        self._create_outfile()

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

            self._extract(self.outfile, dirs)

        if self.rank == 0:
            for i in range(n_dirs):
                self._extract(self.outfile, self.dirs[i])

        self.log.info('Time taken: %s (proc %s)', MPI.Wtime() - start, self.rank)

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

    def _extract(self, d, parallel=True):
        self.log.info('dir: %s (proc %s)'.format(d, self.rank))

        # somehow, it didn't seem possible to have a file open in parallel mode and perform the
        # scatter operation
        out = Dataset(self.outfile, 'a', format='NETCDF4', parallel=parallel)
        if parallel:
            out['XTIME'].set_collective(True)
            if self.lead_day == -1:
                out['start'].set_collective(True)
            for v in self.var_list:
                out[v].set_collective(True)

        with MFDataset(pa.join(d, self.file_glob)) as ds:
            xtime = ds['XTIME']
            xt = np.array(num2date(xtime[:], xtime.units), dtype='datetime64[m]') + pd.Timedelta(self.utc_delta)
            t = date2num(xt.astype(datetime), units=self.time_units)
            if self.lead_day == -1:
                out['start'][self.start + self.rank] = t.min()
                out['XTIME'][self.start + self.rank, :xt.size] = t
            else:
                idx = (xt - xt.min()).astype('timedelta64[D]').astype(int) == self.lead_day
                n = idx.astype(int).sum()
                xout = out['XTIME']
                t_slice = slice(xout.size + self.rank * n, xout.size + (self.rank + 1) * n)
                xout[t_slice] = t[idx]

            for i, v in enumerate(self.var_list):
                var = ds[v]
                x = var[:]
                dims = var.dimensions
                if self.interpolate:
                    x, dims = self._interpolator.netcdf(var)
                if self.lead_day == -1:
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
    def _dims_and_coords(self, ds):
        coords = set.union(*[set(ds[v].coordinates.split()) for v in self.var_list]) - {'XTIME'}
        dims = set.union(*[set(ds[v].dimensions) for v in self.var_list]) - {'Time'}

        def intp(dim):
            return (not self.interpolate or (dim not in self._interpolator.spatial_dims))

        D, C, V = [], [], []
        for d in dims:
            if intp(d):
                D.append((d, ds.dimensions[d].size))

        for c in coords:
            d = ds[c].dimensions[1:]
            if (not self.interpolate) or (len(set(d).intersection(self._interpolator.spatial_dims)) == 0):
                C.append((c, ds[c].dtype, d))

        for v in self.var_list:
            dims = [d for d in ds[v].dimensions if intp(d)]
            if self.interpolate:
                dims.insert(0, 'station')
            V.append((v, ds[v].dtype, dims))

        if self.interpolate:
            D.insert(0, ('station', self.n_stations))
            C.insert(0, ('station', str, 'station'))

        return D, C, V

    # file has to be created and opened BY ALL PROCESSES
    def _create_outfile(self):
        # I think there's a file formate issue - I can open the WRF files only in single-proc mode
        if self.rank == 0:
            ds = Dataset(self._first)
            dcv = self._dims_and_coords(ds)
        else:
            dcv = None

        dcv = self.comm.bcast(dcv, root=0)
        dims, coords, variables = dcv

        out = Dataset(self.outfile, 'w', format='NETCDF4', parallel=True)

        out.createDimension('Time', None)
        if self.lead_day == -1:
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
            out.createVariable(*v[:2], np.r_[['start'], v[2]] if self.lead_day == -1 else v[2])

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
        if self.interpolate and self.rank == 0:
            with Dataset(self.outfile, 'a') as out:
                out['station'][:] = self._names

    @property
    def lonlat(self):
        try:
            return self._lonlat
        except AttributeError:
            # HDFStore can only be opened by one process
            if self.rank == 0:
                sta = self.stations
                lon, lat = sta[['lon', 'lat']].as_matrix().T
                self.n_stations = lon.size
                self._names = sta.index.values
                ll = {'lon': lon, 'lat': lat}
            else:
                ll = None
            self._lonlat = self.comm.bcast(ll, root=0)
            return self._lonlat
