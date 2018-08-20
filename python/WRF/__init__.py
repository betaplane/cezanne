"""
.. automodule:: python.WRF.WRF_threaded
    :members:

.. automodule:: python.WRF.WRF_mpi
    :members:

.. automodule:: python.WRF.tests
    :members:

.. _netCDF4: http://unidata.github.io/netcdf4-python/

"""
import pandas as pd
import numpy as np
import os
from glob import glob
from traitlets.config import Application
from traitlets import List, Integer, Unicode
from functools import partial
from importlib import import_module


def align_stations(wrf, df):
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.Downloader`) with a concatenated (and interpolated to station locations) netCDF file as produced by :class:`.Concatenator`. For now, works with a single field.

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
    """Class which mimics the way :class:`.Concatenator` retrieves the files to concatenate given the :class:`traitlets.config.Application` configuration.

    """

    paths = List(help="list of paths where WRFOUT files can be found").tag(config = True)
    directory_pattern = Unicode('c01_*', help="").tag(config = True)
    wrfout_prefix = Unicode('wrfout').tag(config = True)

    aliases = {'m': 'Meta.file_name'}

    def __init__(self, *args, paths=None, hour=None, from_date=None, pattern=None, limit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

        if paths is not None:
            self.paths = [p for p in paths if os.path.isdir(p)]

        dirs = []
        for p in self.paths:
            if not os.path.isdir(p): continue
            for d in sorted(glob(os.path.join(p, self.directory_pattern if pattern is None else pattern))):
                if (os.path.isdir(d) and not os.path.islink(d)):
                    dirs.append(d)
                    if limit is not None and len(dirs) == limit:
                        break
        dirs = dirs if hour is None else [d for d in dirs if d[-2:] == '{:02}'.format(hour)]
        self.dirs = dirs if from_date is None else [d for d in dirs if d[-10:-2] >= from_date]

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
