#!/usr/bin/env python
import glob
from netCDF4 import Dataset
from datetime import datetime


nc = Dataset('xtime.nc','w')
nc.createDimension('Time',None)
xt = nc.createVariable('XTIME','<f4',('Time'))

files = sorted(glob.glob('/sata1_ceazalabs/carlo/WRFOUT_OPERACIONAL/*00/*d03*'))[::6]

n = Dataset(files.pop(0))
t = n.variables['XTIME']
S = datetime.strptime(t.units[14:],'%Y-%m-%d %H:%M:%S')

for a in t.ncattrs():
	setattr(xt,a,getattr(t,a))
xt[:] = t[:]

for f in files:
	print(f)
	n = Dataset(f)
	t = n.variables['XTIME']
	dt = datetime.strptime(t.units[14:],'%Y-%m-%d %H:%M:%S') - S
	xt[len(xt):len(xt)+len(t)] = t[:] + dt.days*24*60 + dt.seconds/60

nc.close()