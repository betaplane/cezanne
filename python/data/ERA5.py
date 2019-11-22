import pandas as pd
import numpy as np
import os
import cdsapi

c = cdsapi.Client()

years = np.arange(1979, 2020).astype(str)
months = ['{:02d}'.format(m) for m in range(1, 13)]
days = ['{:02d}'.format(d) for d in range(1, 32)]

def get(year, level, directory, name, **kwargs):
    fn = os.path.join(directory, '{}_{}.grb'.format(name, year))
    params = {
        'product_type':'reanalysis',
        'year':[year], 
        'month': months,
        'day': days,
        'time':['00:00', '12:00'],
        'format':'grib',
        'area'  : [0, -160, -70, -60], # North, West, South, East. Default: global
    }
    params.update(kwargs)
    c.retrieve('reanalysis-era5-{}-levels'.format(level), params, fn)

for y in years:
    get(y, 'pressure', '/nfs/HPC/arno/data/reanalyses/ERA5', 'Z500', variable='geopotential', pressure_level='500')
