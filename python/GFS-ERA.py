#!/usr/bin/env python
import iris
from iris.experimental.equalise_cubes import equalise_attributes
from warnings import catch_warnings, simplefilter

iris.FUTURE.netcdf_promote = True

with catch_warnings():
    simplefilter('ignore')
    l1 = iris.load('../../data/fnl/ppt/*00.f06*', 'Total precipitation')
    l2 = iris.load('../../data/fnl/ppt/*12.f06*', 'Total precipitation')

def concat(l):
    equalise_attributes(l)
    c = l.concatenate_cube()
    return c.regrid(a, iris.analysis.Linear(extrapolation_mode='mask'))

c1, c2 = map(concat, [l1, l2])

era = iris.load('../../data/analyses/ERA-ppt.nc')
a = era[0]
em = a.collapsed('time', iris.analysis.MEAN)
g1 = c1.collapsed('initial time', iris.analysis.MEAN)
g2 = c2.collapsed('initial time', iris.analysis.MEAN)
em.remove_coord('time')
g1.remove_coord('initial time')
g2.remove_coord('initial time')
g1.units = 'm'
g2.units = 'm'
