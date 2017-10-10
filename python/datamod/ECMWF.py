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
# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "228", # ppt
#     'dataset'   : "interim",
#     'step'      : "12", # ppt is accumulated, so we request 12h forcast at 00h and 12h to cover whole day
#     'grid'      : "0.75/0.75",
#     'area'      : "10/-180/-90/-60",
#     'time'      : "00/12",
#     'date'      : "1979-01-01/to/2016-12-31",
#     'type'      : "fc", # forecast
#     'class'     : "ei",
#     'format'    : "netcdf",
#     'target'    : "ERA-ppt.nc"
# })

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

# Antarctica
# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "228", # ppt
#     'dataset'   : "interim",
#     'step'      : "12", # ppt is accumulated, so we request 12h forcast at 00h and 12h to cover whole day
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180", # North West South East
#     'time'      : "00/12",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "fc", # forecast
#     'class'     : "ei",
#     'target'    : "ERA-ppt-SH.grb"
# })

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "pl",
#     'levelist'  : "500",
#     'param'     : "129", # geopotential height
#     'dataset'   : "interim",
#     'step'      : "0",
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180", # North West South East
#     'time'      : "00/06/12/18",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "an",
#     'class'     : "ei",
#     'target'    : "ERA-GP500-SH.grb"
# })

# for p, n in [('165', 'U'), ('166', 'V')]:
#     server.retrieve({
#         'stream'    : "oper",
#         'levtype'   : "sfc",
#         'param'     : p,
#         'dataset'   : "interim",
#         'step'      : "0",
#         'grid'      : "0.75/0.75",
#         'area'      : "0/-180/-90/180", # North West South East
#         'time'      : "00/06/12/18",
#         'date'      : "1979-01-01/to/2017-08-31",
#         'type'      : "an",
#         'class'     : "ei",
#         'target'    : "ERA-{}-SH.grb".format(n)
#     })

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "182", # evaporation
#     'dataset'   : "interim",
#     'step'      : "12",
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180", # North West South East
#     'time'      : "00/12",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "fc", # forecast
#     'class'     : "ei",
#     'target'    : "ERA-eva-SH.grb"
# })


# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "84.162", # vertical integral of moisture divergence
#     'dataset'   : "interim",
#     'step'      : "0",
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180", # North West South East
#     'time'      : "00/06/12/18",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "an",
#     'class'     : "ei",
#     'target'    : "ERA-mdiv-SH.grb"
# })

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "167/31.128/84.162", # 2m temperature, sea ice, vert int of moist div 
#     'dataset'   : "interim",
#     'step'      : "0",
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180",
#     'time'      : "00/06/12/18",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "an", # analysis
#     'class'     : "ei",
#     'target'    : "ERA-T2_ice_mdiv-SH.grb"
# })

# server.retrieve({
#     'stream'    : "oper",
#     'levtype'   : "sfc",
#     'param'     : "232.128", # instantaneous moisture flus
#     'dataset'   : "interim",
#     'step'      : "12",
#     'grid'      : "0.75/0.75",
#     'area'      : "0/-180/-90/180", # North West South East
#     'time'      : "00/12",
#     'date'      : "1979-01-01/to/2017-08-31",
#     'type'      : "fc",
#     'class'     : "ei",
#     'target'    : "ERA-mflux-SH.grb"
# })

server.retrieve({
    'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "134.128", # surface pressure
    'dataset'   : "interim",
    'step'      : "0",
    'grid'      : "0.75/0.75",
    'area'      : "0/-180/-90/180",
    'time'      : "00/06/12/18",
    'date'      : "1979-01-01/to/2017-08-31",
    'type'      : "an", # analysis
    'class'     : "ei",
    'target'    : "ERA-SP-SH.grb"
})
# https://stream.ecmwf.int/data/atls02/data/data01/scratch/_mars-atls02-95e2cf679cd58ee9b4db4dd119a05a8d-1lNib4.grib
