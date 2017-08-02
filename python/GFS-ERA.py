#!/usr/bin/env python
import iris
from iris.experimental.equalise_cubes import equalise_attributes
from warnings import catch_warnings, simplefilter
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import iris.coord_categorisation
from datetime import datetime


iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True

# with catch_warnings():
#     simplefilter('ignore')
#     l = iris.load('../../data/fnl/ppt/*f06*', 'Total precipitation')

def concat(l, mean=True, regrid=None, units=None):
    equalise_attributes(l)
    c = l.concatenate_cube()
    if mean:
        c = c.collapsed('initial time', iris.analysis.MEAN)
        c.remove_coord('initial time')
    if regrid is not None:
        c = c.regrid(regrid, iris.analysis.Linear(extrapolation_mode='mask'))
    if units is not None:
        c.units = units
    return c

# c1, c2 = [concat(l, units='mm') for l in [l1, l2]]

a = iris.load('../../data/analyses/ERA-ppt.nc',
              iris.Constraint(time = lambda t: t >= datetime(2015, 7, 8)))[0]
b = a.collapsed('time', iris.analysis.MEAN)
b.remove_coord('time')

def daily_sum(cube):
    iris.coord_categorisation.add_day_of_year(cube, 'time', 'day')
    iris.coord_categorisation.add_year(cube, 'time', 'year')
    c = cube.aggregated_by(['year', 'day'], iris.analysis.SUM).collapsed('time', iris.analysis.MEAN)
    c.remove_coord('time')
    return c


c = iris.load('../../data/fnl/ppt/gdas1.fnl0p25.f06.nc', 'Total precipitation')[0]
# c0 = {i: c.extract(iris.Constraint(time = iris.time.PartialDateTime(hour = i))) for i in [0, 6, 12, 18]}
cs = c.aggregated_by('day', iris.analysis.SUM)

iris.coord_categorisation.add_hour(c, 'time')
h = c.aggregated_by('hour', iris.analysis.MEAN)
h0 = h.extract(iris.Constraint(hour = lambda t: t<7)).collapsed('time', iris.analysis.MEAN)
h1 = h.extract(iris.Constraint(hour = lambda t: t>11)).collapsed('time', iris.analysis.MEAN)

c0 = c.extract(iris.Constraint(time = lambda t: t<datetime(2016, 4, 25))).collapsed('time', iris.analysis.MEAN)
c1 = c.extract(iris.Constraint(time = lambda t: t>=datetime(2016, 4, 25))).collapsed('time', iris.analysis.MEAN)

