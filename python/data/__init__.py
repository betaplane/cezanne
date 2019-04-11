"""
.. automodule:: python.data.CEAZA
    :members:

.. automodule:: python.data.compare
    :members:

.. automodule:: python.data.interpolate
    :members:

.. automodule:: python.data.IGRA
    :members:

.. automodule:: python.data.NCDC
    :members:

.. automodule:: python.data.EarthData
    :members:

.. automodule:: python.data.IceBridge
    :members:

.. automodule:: python.data.GDAL
    :members:

.. _netCDF4: http://unidata.github.io/netcdf4-python/

"""
import os
from traitlets.config import Application, Configurable, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode
from glob import glob

class MixIn(object):
    def glob(self, pattern):
        g = glob(pattern)
        if len(g) == 0:
            g = glob(os.path.join(self.path, pattern))
        assert len(g) > 0, "No files matching the glob pattern found."
        return g

class DataApp(MixIn, Application):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    path = Unicode().tag(config=True)
    def __init__(self, *args, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound:
            cfg = Config(config)
        super().__init__(config=cfg, **kwargs)

class DataConf(MixIn, Configurable):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    path = Unicode().tag(config=True)
    def __init__(self, *args, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            super().__init__(config=cfg, **kwargs)
        except ConfigFileNotFound:
            super().__init__(**kwargs)
