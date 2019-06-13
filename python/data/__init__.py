"""
.. automodule:: data.CEAZA
    :members:

.. automodule:: data.WRF
    :members:

.. automodule:: data.compare
    :members:

.. automodule:: data.interpolate
    :members:

.. automodule:: data.IGRA
    :members:

.. automodule:: data.NCDC
    :members:

.. automodule:: data.EarthData
    :members:

.. automodule:: data.IceBridge
    :members:

.. automodule:: data.GDAL
    :members:

.. automodule:: data.GSHHS
    :members:

.. _netCDF4: http://unidata.github.io/netcdf4-python/

"""
import os
from glob import glob
from cezanne import App, Conf, Unicode

class MixIn(object):
    def glob(self, pattern):
        g = glob(pattern)
        if len(g) == 0:
            g = glob(os.path.join(self.path, pattern))
        assert len(g) > 0, "No files matching the glob pattern found."
        return g

    @property
    def full_path(self):
        return os.path.join(self.path, self.file_name)

class DataApp(MixIn, App):
    path = Unicode().tag(config=True)

class DataConf(MixIn, Conf):
    path = Unicode().tag(config=True)
