#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

# https://www.ecmwf.int/en/research/climate-reanalysis/browse-reanalysis-datasets
# https://software.ecmwf.int/wiki/display/WEBAPI/Python+ERA-interim+examples
# https://software.ecmwf.int/wiki/display/UDOC/MARS+keywords
# http://apps.ecmwf.int/codes/grib/param-db

server = ECMWFDataServer()

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "167", # 2m temperature
#     'dataset'   : "interim",
#     'step'      : "0",
#     'grid'      : "0.75/0.75",
#     'area'      : "10/-180/-90/-60",
#     'time'      : "00/06/12/18",
#     'date'      : "1979-01-01/to/2016-12-31",
#     'type'      : "an", # analysis
#     'class'     : "ei",
#     'format'    : "netcdf",
#     'target'    : "ERA-T2.nc"
# })

# https://software.ecmwf.int/wiki/pages/viewpage.action?pageId=56658233
server.retrieve({
    'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "228", # ppt
    'dataset'   : "interim",
    'step'      : "12", # ppt is accumulated, so we request 12h forcast at 00h and 12h to cover whole day
    'grid'      : "0.75/0.75",
    'area'      : "10/-180/-90/-60",
    'time'      : "00/12",
    'date'      : "1979-01-01/to/2016-12-31",
    'type'      : "fc", # forecast
    'class'     : "ei",
    'format'    : "netcdf",
    'target'    : "ERA-ppt.nc"
})

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "pl",
#     'levelist'  : "850",
#     'param'     : "131/132", 
#     'dataset'   : "interim",
#     'step'      : "0",
#     'grid'      : "0.75/0.75",
#     'area'      : "10/-180/-90/-60",
#     'time'      : "00/06/12/18",
#     'date'      : "1979-01-01/to/2016-12-31",
#     'type'      : "an",
#     'class'     : "ei",
#     'format'    : "netcdf",
#     'target'    : "ERA-interim.nc"
# })
