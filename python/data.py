#!/usr/bin/env python
from glob import glob
import os
from configparser import ConfigParser
import xarray as xr


conf = ConfigParser()
conf.read('data.cfg')


for f in glob(os.path.join(conf['datadir']['wrf'], '*.nc')):
    nc = xr.open_dataset(f)
