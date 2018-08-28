from zipfile import ZipFile
from cartopy.io.shapereader import Reader, GEOMETRY_FACTORIES
from shapefile import Reader as shapereader
from traitlets import Unicode
from traitlets.config.configurable import Configurable
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
import os


class GSHHS(Configurable, Reader):
    """Loader for the Global Self-consistent, Hierarchical, High-resolution Geography Database (GSHHG). Works from directly from the downloaded zipfile. For use examples, see :class:`plots.Coquimbo`.

    """
    path = Unicode('').tag(config = True)

    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)

    def __init__(self, filename, *args, config={}, **kwargs):
        try:
            config = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            config.merge(config)
        except ConfigFileNotFound: pass
        super().__init__(config=config, **kwargs)

        z = ZipFile(self.path)

        self._reader = shapereader(
            **{k: z.open('{}.{}'.format(filename, k)) for k in ['shp', 'shx', 'dbf']})
        self._geometry_factory = GEOMETRY_FACTORIES.get(self._reader.shapeType)
        self._fields = self._reader.fields
        z.close()
