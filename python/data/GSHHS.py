"""
GSHHS
=====

"""
from zipfile import ZipFile
from cartopy.io.shapereader import Reader, GEOMETRY_FACTORIES
from shapefile import Reader as shapereader
from traitlets import Unicode
from traitlets.config.configurable import Configurable
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
import os

# there's a problem with newer sphinx and multiple imports. apparently solved only for python 3.7
# https://github.com/sphinx-doc/sphinx/pull/5998
# https://github.com/ederag/geoptics/pull/2
class GSHHS(Configurable, Reader):
    """Loader for the Global Self-consistent, Hierarchical, High-resolution Geography Database (GSHHG). Works from directly from the downloaded zipfile. For use examples, see :class:`plots.Coquimbo`.

    """
    path = Unicode('').tag(config = True)

    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)

    def __init__(self, filename, *args, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound: pass
        super().__init__(config=cfg, **kwargs)

        z = ZipFile(self.path)

        self._reader = shapereader(
            **{k: z.open('{}.{}'.format(filename, k)) for k in ['shp', 'shx', 'dbf']})
        self._geometry_factory = GEOMETRY_FACTORIES.get(self._reader.shapeType)
        self._fields = self._reader.fields
        z.close()

    def clip(self, bbox):
        """Return all geometries in the reader which intersect a given bounding box.

        :param bbox: bounding box as (minx, maxx, miny, maxy)
        :type bbox: :obj:`iterable`
        :returns: collection of :mod:`shapely.geometry` objects
        :rtype: :obj:`list`

        """
        from shapely.geometry import Polygon, LinearRing
        poly = Polygon(LinearRing(zip(bbox[[0, 1, 1, 0]], bbox[[2, 2, 3, 3]])))
        return [g for g in self.geometries() if poly.intersects(g)]

    @classmethod
    def stadtlandfluss(cls, bbox=None):
        """Convenience wrapper to return a named tuple with ``coast``, ``border`` and ``rivers`` shapes, optionally :meth:`clipped <clip>` to a bounding box.

        :param bbox: bounding box for :meth:`clip`
        :returns: named tuple with ``coast``, ``border`` and ``rivers``
        :rtype: :class:`~collections.namedtuple`

        """
        from collections import namedtuple
        t = namedtuple('GSHHS', ['coast', 'border', 'rivers'])
        coast = cls('GSHHS_shp/i/GSHHS_i_L1')
        border = cls('WDBII_shp/i/WDBII_border_i_L1')
        rivers = cls('WDBII_shp/i/WDBII_river_i_L05')
        result = t(coast, border, rivers)
        return result if bbox is None else t(*[g.clip(bbox) for g in result])
