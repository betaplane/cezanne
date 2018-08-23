"""
.. _netCDF4: http://unidata.github.io/netcdf4-python/

.. automodule:: python.WRF.threads
    :members:

.. automodule:: python.WRF.MPI
    :members:

.. automodule:: python.WRF.tests
    :members:

"""
import pandas as pd
import numpy as np
import os
from glob import glob
from traitlets.config import Application
from traitlets import List, Integer, Unicode, Bool, Instance
from functools import partial
from importlib import import_module


def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.CEAZAMet`) with a concatenated (and interpolated to station locations) netCDF file as produced by this packages concatenators (:class:`threads.Concatenator`, :class:`MPI.Concatenator`). For now, works with a single field.

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

class WRFiles(Application):
    """Base class for the WRF-file concatenators in this package (:mod:`.threads` and :mod:`.MPI`). It retrieves the original WRF output files and is configured/run via the :class:`traitlets.config.Application`  interface.

    """

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

    aliases = {'m': 'Meta.file_name'}

    def __init__(self, *args, limit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

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
        self.log.info("WRFiles initialized with %s directories.", len(self.dirs))

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
        """Get the first netCDF file matching the given arguments (see :class:`Concatenator` for a description), based on the configuration values (section *wrfout*) in the global config file.

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
