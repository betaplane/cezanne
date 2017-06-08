#!/usr/bin/python
import gdal, os, re
from glob import glob
from datetime import datetime
import pandas as pd
import numpy as np

dir = '../../data/MODIS'

def array(f, value):
    # d = datetime.strptime(re.search('A(\d*)_', f).group(1), '%Y%m%d')
    ds = gdal.Open(f)
    return pd.DataFrame(ds.GetRasterBand(1).ReadAsArray()).replace(value, np.nan)

a = pd.Panel(dict([(i, array(f, 250)) for i, f in enumerate(glob(os.path.join(dir, '*.tif')))]))

b = a.fillna(method='ffill', axis=0)
c = a.fillna(method='bfill', axis=0)
