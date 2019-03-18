from osgeo import gdal
from cartopy import crs
import numpy as np
import xarray as xr
import re

class Dataset(object):
    def __init__(self, filename):
        self._ds = gdal.Open(filename)
        bands = [self._ds.GetRasterBand(i+1) for i in range(self._ds.RasterCount)]
        self._bands = {b.GetDescription(): b for b in bands}
        self.bands = self._bands.keys()
        self._tr = np.reshape(self._ds.GetGeoTransform(), (2, 3))
        self.proj = crs.epsg(int(re.search('(\d*).\]\]$', self._ds.GetProjection()).group(1)))

    def __getitem__(self, descr):
        x = self._bands[descr].ReadAsArray()
        r, c = np.mgrid[:x.shape[0], :x.shape[1]]
        # important!: reverse order of r(ow), c(olumn)
        # https://www.gdal.org/gdal_datamodel.html
        ij1 = np.vstack((np.ones(np.prod(x.shape)), c.flatten(), r.flatten()))
        i, j = [np.reshape(k, x.shape) for k in self._tr.dot(ij1)]
        return xr.Dataset({k: xr.DataArray(v, dims=['i', 'j']) for k, v in {descr: x, 'SN': j, 'WE': i}.items()})
