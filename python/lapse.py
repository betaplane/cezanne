#!/usr/bin/env python
import numpy as np
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams


K = 273.15
dd = lambda s: os.path.join('../data',s)

nc = Dataset(dd('wrf/d02_2014-09-10_transf.nc'))

t = nc.variables['temp'][:]
g = nc.variables['ghgt'][:]
b = nc.variables['PBLH'][:]
z = nc.variables['HGT'][:]
# h = g - z
# b4d = np.r_['0,4',b].repeat(29,0).transpose([1,0,2,3])
# n = np.sum(h<b4d, 1)

dg = np.diff(g,1,1)
dt = np.diff(t,1,1)
lr = dt / dg * 1000
lrs = np.nanstd(lr,0)

fig = plt.figure(subplotpars=SubplotParams(left=.02,right=.98,bottom=.02,top=.98))
plt.set_cmap('viridis')
# plt.set_cmap('coolwarm')
for i in range(28):
	ax = plt.subplot(4,7,i+1)
	pl = plt.pcolormesh(lrs[i,:,:])
# 	pl.set_clim(np.array([-1,1])*max(np.abs(pl.get_clim())))
	cb = plt.colorbar()
	ax.set_xticklabels([])
	ax.set_yticklabels([])
