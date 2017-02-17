#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def cut(x,y,nc,d=3):
	i,j = nc.I_PARENT_START, nc.J_PARENT_START
	s = y.shape
	return x.take(range(i-1,int(i-1+s[-1]/3)),-1).take(range(j-1,int(j-1+s[-2]/3)),-2)
	
def down(x,d=3):
	def m(x):
		return x.reshape((x.shape[0],-1,d)).mean(2).reshape((x.shape[0],-1,int(x.shape[2]/d)))
	def t(x):
		return x.transpose((0,2,1))
	return t(m(t(m(x))))

def up(x,d=3):
	return x.repeat(3,-2).repeat(3,-1)

if __name__ == '__main__':
	np = Dataset('wrfout_d02_2015-12-31_00_pablo.nc')
	n2 = Dataset('wrfout_d02_2015-12-31_00_op.nc')
	n3 = Dataset('wrfout_d03_2015-12-31_00_op.nc')
	t3 = n3.variables['HFX'][:]
	t2 = n2.variables['HFX'][:]

	t2c = cut(t2,t3,n3)
	t3d = down(t3)
	dtd = t3d-t2c
	dtu = t3-up(t2c)
	dtt = t3-up(t3d)
	
	fig = plt.figure()
	plt.pcolormesh(dtd[0,:,:])
	plt.colorbar()
	fig.show()