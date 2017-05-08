#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from astropy.stats import LombScargle
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.figure import SubplotParams
import matplotlib.gridspec as gs
import helpers as hh
from functools import partial
import stipolate as st
import mapping as mp


K = 273.15
dd = lambda s: os.path.join('../../data/tables',s)

D = pd.HDFStore(dd('station_data.h5'))

sta = D['sta']
T = hh.extract(D['ta_c'],'prom') + K

S = pd.HDFStore(dd('LinearLinear.h5'))
Tm = S['T2n']


def t(d):
	return np.array(d.index,dtype='datetime64[h]').astype(float)

def xx(ax):
	m = (lambda x:(min(x[:,0]),max(x[:,1])))( np.array([x.get_xlim() for x in ax]) )
	for x in ax:
		x.set_xlim(m)
	return m
	
def yy(ax):
	y = np.array([x.get_ylim() for x in ax])
	d = np.diff(y,1,1)
	m = max(d)
	y = y[:,:1] + d/2
	y = np.r_['1',y-m/2,y+m/2]
	for i,x in enumerate(ax):
		x.set_ylim(y[i,:])
	return m

def power(d,f):
	return (lambda c:LombScargle(t(c),c).power(f,normalization='psd')/len(c))(d.dropna())

def model(d,f):
	i = d.asfreq('1H')
	return pd.DataFrame(LombScargle(t(d),d).model(t(i),f),index=i.index)
	
def annual(s):
	fig=plt.figure(figsize=(8,8))
	grid = gs.GridSpec(5,5)
	tx = lambda x:(x.index,x)
	
	ax = []
	ay = []	
	for j in range(5):
		D=T if j==0 else Tm.minor_xs(Tm.minor_axis[j-1])
		i, y = model(D[s].dropna(),1/(24*365.25))
		
		ax.append(plt.subplot(grid[j,:4]))
		plt.plot_date(c.index,c,'-')
		plt.plot_date(y.index,y,'-')
		ax[-1].grid()
		
		ay.append(plt.subplot(grid[j,4]))
		plt.plot(*tx(c.groupby(c.index.month).mean()))
		plt.plot(*tx(y.groupby(y.index.month).mean()))
		ay[-1].set_xticks(range(3,13,3))
		ay[-1].yaxis.set_ticks_position('right')
		ay[-1].grid()
		if j<4:
			ax[-1].set_xticklabels([])
			ay[-1].set_xticklabels([])
	fig.show()
	
	xx(ax)
	yy(ax)
	yy(ay)
	return ax,ay
		
# ax1,ax2 = cyp('LAGEL')

def daily(s):
	fig=plt.figure()
	x = T[s].dropna()
	for l in ['d03_op', 'd03_orl']:
		y = Tm[s][l]-x
		d = y.groupby(y.index.hour).mean().dropna()
		plt.plot(d.index,d,'-',label=l)
	plt.legend()
	fig.show()

g2 = Dataset(dd('../WRF/3d/grid_d03.nc'))
Map = mp.basemap(g2)

n0 = Dataset(dd('../WRF/3d/d03_day0.nc'))
x0 = st.interp_nc(n0,'T2',sta,method='linear', map=Map)
x0_1 = x0.iloc[:7440,:]
x0_2 = x0.iloc[7440:,:]

n4 = Dataset(dd('../WRF/3d/T2_4.nc'))
x4 = st.interp_nc(n4,'T2',sta,method='linear', map=Map)
x4_1 = x4.iloc[:7440,:]
x4_2 = x4.iloc[7440:,:]

fig = plt.figure()
for i,x in enumerate([Tm['d03_orl'],x0_1,x0_2,x4_1,x4_2]):
	# y = (x-T)
	y = (x-T)**2
	# y = (y-y.mean())
	y = y.groupby(y.index.hour).mean().mean(1)
	plt.plot(y,label=['orl','op+0/00','op+0/12','op+4/00','op+4/12'][i])
plt.legend()


