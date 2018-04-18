# https://hdfeos.org/software/pyhdf.php
from pyhdf import SD
import xarray as xr
import re, os, requests

filename = '../../../data/MODIS/cloud/MYD08_D3.A2015001.061.2018047212334.hdf'
base_url = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/501227680'

class hdf(object):
    def __init__(self, filename=None, copy=None):
        if copy is None:
            self.sd = SD.SD(filename)
            self.x = self.sd.select('XDim')[:]
            self.y = self.sd.select('YDim')[:]
            self.filename = filename
        else:
            for a in ['sd', 'x', 'y']:
                setattr(self, a, getattr(copy, a))

    def __del__(self):
        self.sd.end()

    def list(self, pattern):
        d = self.sd.datasets()
        return [k for k in d.keys() if re.search(pattern, k, re.IGNORECASE)]

    def to_netcdf(self, dataset):
        x = xr.DataArray(self.sd.select(dataset)[:], coords=[('y', self.y), ('x', self.x)])
        xr.Dataset({dataset: x}).to_netcdf('{}.nc'.format(os.path.splitext(self.filename)[0]))

