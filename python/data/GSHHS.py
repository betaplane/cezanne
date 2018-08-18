from zipfile import ZipFile
from cartopy.io.shapereader import Reader, GEOMETRY_FACTORIES
from shapefile import Reader as shapereader
from traitlets import Unicode
from traitlets.config.configurable import Configurable
import os

class GSHHS_Reader(Configurable, Reader):
    path = Unicode('').tag(config = True)

    def __init__(self, filename, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

        if path is not None:
            self.path = path
        z = ZipFile(self.path)

        self._reader = shapereader(
            **{k: z.open('{}.{}'.format(filename, k)) for k in ['shp', 'shx', 'dbf']})
        self._geometry_factory = GEOMETRY_FACTORIES.get(self._reader.shapeType)
        self._fields = self._reader.fields
        z.close()
