import gdal, re
import xarray as xr
import numpy as np

def tif2xr(filename):
    ds = gdal.Open(filename)
    A = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetGeoTransform()
    lon = (np.arange(ds.RasterXSize) + .5) * g[1] + g[0]
    lat = (np.arange(ds.RasterYSize) + .5) * g[5] + g[0]
    x = xr.DataArray(A, coords=[('lat', lat), ('lon', lon)])
    date = re.search('\.A(\d{7})', filename).group(1)
    t = np.datetime64('{}-01-01'.format(date[:4])) + np.timedelta64(int(date[4:]) - 1, 'D')
    X = x.expand_dims('time')
    X['time'] = ('time', [t])
    return X
