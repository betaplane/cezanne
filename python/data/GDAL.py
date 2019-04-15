"""
Raster Datasets
---------------

.. _supported_rasters:

Currently supported raster formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `GDAL <https://www.gdal.org/>`_
* :mod:`rasterio`
* `pysheds <https://github.com/mdbartos/pysheds>`_ (uses rasterio internally, but its 'grid' class has slightly different attribute names from raw rasterio)

"""
from importlib import import_module
from functools import singledispatch
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

        * Works only for the one projection I have used so far...
    """
    crs = import_module('cartopy.crs')
    d = dict(filter(lambda i: i is not None, map(proj4_parser, proj4.split())))
    return getattr(crs, proj4_cartopy_projections[d.pop('proj')])(d)

def wkt2proj(obj):
    """Map wkt string to Proj.4 string. Argument can be either the wkt string **or** a '.prj' filename.

    """
    osr = import_module('osgeo.osr')
    try:
        with open(obj) as f:
            s = f.read()
    except FileNotFoundError:
        s = obj
    return osr.SpatialReference(s).ExportToProj4()

def coords(raster, latlon=False):
    """Return coordinate arrays constructed from a raster dataset's affine transform.

    :param raster: GDAL or rasterio (including pysheds) raster dataset
    :param latlon: If `True`, return lon, lat coordinates, otherwise projected coordinates
    :returns: (lon, lat) or (i, j) coordinates corresponding to the raster dataset
    :rtype: :obj:`tuple` of :class:`ndarrays <numpy.ndarray>`

    """
    shape, a, proj = _raster_info(raster, latlon)
    i, j = [k+.5 for k in np.mgrid[:shape[0], :shape[1]]]
    x = a[0] * j + a[1] * i + a[2]
    y = a[3] * j + a[4] * i + a[5]
    return (x, y) if proj is None else proj(x, y, inverse=True)

@singledispatch
def _raster_info(raster, latlon):
    pass

try:
    gdal = import_module('osgeo.gdal')
    @_raster_info.register(gdal.Dataset)
    def _(ds, latlon):
        b = ds.GetRasterBand(1)
        # https://www.gdal.org/gdal_datamodel.html
        # rasterio's affine is switched relative to GDAL's GeoTransform
        a = np.array(ds.GetGeoTransform())[[1, 2, 0, 4, 5, 3]]
        if latlon:
            proj = import_module('pyproj')
            p = proj.Proj(wkt2proj(ds.GetProjection()))
        return ((b.YSize, b.XSize), a, p if latlon else None)
except: pass

try:
    io = import_module('rasterio.io')
    @_raster_info.register(io.DatasetReader)
    def _(ds, latlon):
        if latlon:
            proj = import_module('pyproj')
            p = proj.Proj(ds.crs.to_proj4())
        return (ds.shape, ds.transform, p if latlon else None)
except: pass

try:
    grid = import_module('pysheds.grid')
    @_raster_info.register(grid.Grid)
    def _(ds, latlon):
        return (ds.shape, ds.affine, ds.crs if latlon else None)
except: pass

def mv2nan(data, value, copy=True):
    if copy:
        data = data.copy()
    if value is not None:
        try:
            data[data==value] = np.NAN
        except ValueError:
            data = np.array(data, dtype='float')
            data[data==value] = np.NAN
    return data

class Affine:
    """Wrapper around the type of affine transformations used by raster datasets. Initialize with one of the :ref:`supported raster formats <supported_rasters>`.

    .. TODO::

        * merge with :func:`python.geo.affine`

    """
    def __init__(self, raster):
        self.shape, a, self.proj = _raster_info(raster, latlon=True)
        b = np.reshape(a[:6], (2, 3))
        self.mult = b[:, :2]
        self.add = b[:, 2:]

    def ij(self, x, y, latlon=False):
        """Invert the transform to get raster indexes ('i, j') from coordinates.

        :param x: longitude / projected x-dimension
        :type x: number or :class:`~numpy.ndarray`
        :param y: latitude / projected y-dimension
        :type y: number or :class:`~numpy.ndarray`
        :param latlon: If ``True``, ``x`` and ``y`` are lon, lat coordinates, otherwise projected (in the raster's projection)
        :returns: 'i, j' raster indexes corresponding to the coordinates ``x``, ``y``
        :rtype: :obj:`tuple` of numbers or :class:`arrays <numpy.ndarray>`

        """
        s = np.array(x).shape
        if not projected:
            x, y = self.proj(x, y)
        xy = np.vstack([np.array(i).flatten() for i in (x, y)]) - self.add
        return [np.floor(i).reshape(s).astype(int) for i in np.linalg.inv(self.mult).dot(xy)]


class GeoTiff(object):
    """Wrapper around a `GDAL Dataset <https://www.gdal.org/gdal_datamodel.html>`_, so far used only for GeoTiffs. Init with the filename to be opened, and optional a band number (defaults to 1). Set via :attr:`band_no` to something other than 1 (if desired) **before** using any of the other methods.

    .. attribute:: band_no

        Index (1-based) of the band to operate on. Change directly before using any of the methods if a band different from 1 is desired.

    .. attribute:: proj

        :class:`pyproj.Proj` initialized from the GDAL Dataset's :meth:`GetProjection` method.

    """
    def __init__(self, filename, band=1):
        gdal = import_module('osgeo.gdal')
        osr = import_module('osgeo.osr')
        proj = import_module('pyproj')
        self._ds = gdal.Open(filename)
        self.proj = proj.Proj(osr.SpatialReference(self.wkt).ExportToProj4())
        self.band_no = band

    def coords(self, latlon=False):
        """Return (lon, lat) tuple of the dataset's coordinates.

        :param latlon: if ``True``, return lon/lat coordinates, otherwise projected (as in the original raster)
        :returns: (lon, lat) or (i, j) in projected coordinates
        :rtype: :obj:`tuple`

        """
        return coords(self._ds, latlon)

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


    @property
    def data(self):
        "Data corresponding to band :attr:`band_no` as :class:`numpy.ndarray`, with no-value data replaced by :obj:`numpy.nan` (which means convert the data to float if it isn't already)."
        try:
            return self._data[self.band_no]
        except AttributeError:
            self._data = {self.band_no: mv2nan(self.band.ReadAsArray(), self.band.GetNoDataValue())}
        except KeyError:
            self._data[self.band_no] = mv2nan(self.band.ReadAsArray(), self.band.GetNoDataValue())
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
