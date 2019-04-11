"""
GDAL Wrappers
-------------

"""
from osgeo import gdal, osr
from pyproj import Proj
from importlib import import_module
import numpy as np
import re

# http://proj.maptools.org/gen_parms.html
proj4_cartopy_projections = {
    'tmerc': 'TransverseMercator'
}

def proj4_parser(proj4):
    """Helper function to parse Proj.4 strings, returns key-value pairs with the leading '+' stripped, and nothing if there's only a key without value.

    """
    try:
        k, v = proj4.split('=')
    except ValueError: return
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError: pass
    return k.replace('+', ''), v

def proj2cartopy(proj4):
    """Return :mod:`cartopy.crs.CRS` projection given a Proj.4 string.

    .. TODO::

        * Works only for projections I have used so far
    """
    crs = import_module('cartopy.crs')
    d = dict(filter(lambda i: i is not None, map(proj4_parser, proj4.split())))
    return getattr(crs, proj4_cartopy_projections[d.pop('proj')])(d)

def wkt2proj(obj):
    """Map wkt string to Proj.4 string. Argument can be either the wkt string **or** a '.prj' filename.

    """
    try:
        with open(obj) as f:
            s = f.read()
    except FileNotFoundError:
        s = obj
    return osr.SpatialReference(s).ExportToProj4()



class GeoTiff(object):
    """Wrapper around a `GDAL Dataset <https://www.gdal.org/gdal_datamodel.html>`_, so far used only for GeoTiffs. Init with the filename to be opened, and optional a band number (defaults to 1). Set via :attr:`band_no` to something other than 1 (if desired) **before** using any of the other methods.

    .. attribute:: band_no

        Index (1-based) of the band to operate on. Change directly before using any of the methods if a band different from 1 is desired.

    .. attribute:: proj

        :class:`pyproj.Proj` initialized from the GDAL Dataset's :meth:`GetProjection` method.

    """
    def __init__(self, filename, band=1):
        self._ds = gdal.Open(filename)
        self.proj = Proj(osr.SpatialReference(self.wkt).ExportToProj4())
        self.band_no = band

    def coords(self, projected=True):
        """Return (lon, lat) tuple of the dataset's coordinates.

        :param projected: if ``True``, return projected coordinates, otherwise lon/lat
        :returns: (lon, lat) or (i, j) in projected coordinates
        :rtype: :obj:`tuple`

        """
        # https://www.gdal.org/gdal_datamodel.html
        i, j = [k+.5 for k in np.mgrid[:self.band.YSize, :self.band.XSize]]
        g = self.GetGeoTransform()
        x = g[0] + g[1] * j + g[2] * i
        y = g[3] + g[4] * j + g[5] * i
        return (x, y) if projected else self.proj(x, y, inverse=True)

    @property
    def wkt(self):
        "WKT string corresponding to dataset's projection."
        return self._ds.GetProjection()

    @property
    def proj4(self):
        "Proj.4 string corresponding to dataset's projection."
        return self.proj.srs

    @property
    def band(self):
        "Currently set band (corresponding to index :attr:`band_no`)."
        try:
            return self._band[self.band_no]
        except AttributeError:
            self._band = {self.band_no: self._ds.GetRasterBand(self.band_no)}
        except KeyError:
            self._band[self.band_no] = self._ds.GetRasterBand(self.band_no)
        return self._band[self.band_no]

    @staticmethod
    def _replace_no_data(data, value):
        if value is not None:
            try:
                data[data==value] = np.NAN
            except ValueError:
                data = np.array(data, dtype='float')
                data[data==value] = np.NAN
        return data

    @property
    def data(self):
        "Data corresponding to band :attr:`band_no` as :class:`numpy.ndarray`, with no-value data replaced by :obj:`numpy.nan` (which means convert the data to float if it isn't already)."
        try:
            return self._data[self.band_no]
        except AttributeError:
            self._data = {self.band_no: self._replace_no_data(self.band.ReadAsArray(), self.band.GetNoDataValue())}
        except KeyError:
            self._data[self.band_no] = self._replace_no_data(self.band.ReadAsArray(), self.band.GetNoDataValue())
        return self._data[self.band_no]

    @property
    def shape(self):
        "data shape"
        return (self.band.YSize, self.band.XSize)

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    @property
    def cartopy(self):
        ":class:`cartopy.crs.CRS` projection corresponding to dataset."
        return proj2cartopy(self.proj4)

    def pcolormesh(self, ax=None, background=None, **kwargs):
        """Produce :meth:`~matplotlib.pyplot.pcolormesh` plot of the dataset.

        :param ax: existing axes or ``None`` (will be created with :func:`~matplotlib.pyplot.axes`)
        :type ax: :class:`cartopy.mpl.geoaxes.GeoAxes`
        :param background: If not ``None``, set background_path and outline_patch to colors which can be specified by passing a :obj:`dict` with values for keys ``patch`` and ``outline``, respectively. The defaults are 'none' and 'w' (transparent and white) and are used if an empte :obj:`dict` (``{}``) is passed.
        :type background: ``None`` or :obj:`dict`
        :returns: plot handle
        :rtype: :class:`~matplotlib.collections.QuadMesh`

        """
        if ax is None:
            plt = import_module('matplotlib.pyplot')
            ax = plt.axes(projection=self.cartopy)
        i, j = self.coords()
        pl = ax.pcolormesh(i, j, self.data, **kwargs)
        if background is not None:
            ax.background_patch.set_color(background.get('patch', 'none'))
            ax.outline_patch.set_edgecolor(background.get('outline', 'w'))
        return pl
