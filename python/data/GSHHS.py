from importlib import import_module
from cartopy.io.shapereader import Reader, GEOMETRY_FACTORIES
from traitlets import Unicode, List
from traitlets.config.configurable import Configurable
import numpy as np
import os

class GSHHS_Reader(Configurable, Reader):
    path = Unicode('').tag(config = True)
    bbox = List([-72.2, -69.8, -32.5, -28.2])

    def __init__(self, filename, path=None):
        loader = import_module('traitlets.config.loader')
        super().__init__(
            config = loader.PyFileConfigLoader(
                os.path.expanduser('~/Dropbox/work/config.py')).load_config()
        )
        if path is not None:
            self.path = path
        zipf = import_module('zipfile')
        z = zipf.ZipFile(self.path)

        reader = import_module('shapefile')
        cart = import_module('cartopy.io.shapereader')
        self._reader = reader.Reader(
            **{k: z.open('{}.{}'.format(filename, k)) for k in ['shp', 'shx', 'dbf']})
        self._geometry_factory = GEOMETRY_FACTORIES.get(self._reader.shapeType)
        self._fields = self._reader.fields
        z.close()

    def clip(self, geom):
        np.array(geom.envelope.bounds) < self.bbox
