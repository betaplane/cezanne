#!/usr/bin/env python
from netCDF4 import Dataset
from urllib.parse import urlencode

base_url = 'http://motherlode.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p25deg/TwoD'

d = {
    'var': 'Total_precipitation_surface_Mixed_intervals_Accumulation',
    'south': -33,
    'north': -28,
    'west': -72,
    'east': -68,
    'disableProjSubset': 'on',
    'horizStride': 1,
    'time_start': '2017-07-29T03:00:00Z',
    'time_end': '2017-08-16T12:00:00Z',
    'timeStride': 1,
    'vertCoord': '',
    'addLatLon': 'true',
    'accept': 'netcdf'
}

p = urlencode(d)

u = 'http://motherlode.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p25deg/TwoD?var=Total_precipitation_surface_Mixed_intervals_Accumulation&north=-28&west=-72&east=-69&south=-33&disableProjSubset=on&horizStride=1&time_start=2017-07-29T03%3A00%3A00Z&time_end=2017-08-16T12%3A00%3A00Z&timeStride=1&vertCoord=&addLatLon=true&accept=netcdf'
