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
from glob import glob
from cezanne import App, Conf

class MixIn(object):
    def glob(self, pattern):
        g = glob(pattern)
        if len(g) == 0:
            g = glob(os.path.join(self.path, pattern))
        assert len(g) > 0, "No files matching the glob pattern found."
        return g

class DataApp(MixIn, App):
    path = Unicode().tag(config=True)

class DataConf(MixIn, Conf):
    path = Unicode().tag(config=True)
