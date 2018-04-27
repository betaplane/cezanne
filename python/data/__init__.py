"""
.. automodule:: python.data.CEAZA
    :members:

.. automodule:: python.data.compare
    :members:

.. automodule:: python.data.interpolate
    :members:

.. automodule:: python.data.WRF
    :members:

.. automodule:: python.data.IGRA
    :members:

.. automodule:: python.data.NCDC
    :members:

"""
from configparser import ConfigParser
config = ConfigParser()
"""global configuration values"""
config.read('/HPC/arno/general.cfg')
