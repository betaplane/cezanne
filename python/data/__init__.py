"""
.. automodule:: python.data.CEAZA
    :members:

.. automodule:: python.data.compare
    :members:

.. automodule:: python.data.interpolate
    :members:

.. automodule:: python.data.WRF
    :members:

.. automodule:: python.data.tests
    :members:

.. automodule:: python.data.IGRA
    :members:

.. automodule:: python.data.NCDC
    :members:

"""
from configparser import ConfigParser
from importlib.util import find_spec
from importlib import import_module

config = ConfigParser()
"""global configuration values"""
config.read('/HPC/arno/general.cfg')

if find_spec('mpi4py') is None:
    WRF = import_module('.WRF_threaded', 'data')
else:
    WRF = import_module('.WRF_mpi', 'data')
