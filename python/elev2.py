#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers as hh
from netCDF4 import Dataset
from scipy.stats import gaussian_kde
from scipy.optimize import brute


D = pd.HDFStore('../data/station_data.h5')
S = pd.HDFStore('../data/LinearLinear.h5')

T = hh.extract(D['ta_c'],'prom',1)
Tm = S['T2']
sta = D['sta']
Z = S['z']
L = S['land_mask']

B = Tm['d02']-T
dz = Z['d02']-sta['elev']
lm = L['d02']

B = Tm['d03_orl']-T
dz = Z['d03_orl']-sta['elev']
lm = L['d03_orl']

# B = Tm['d03_0_12']-T
# dz = Z['d03_op']-sta['elev']
# lm = L['d03_op']
# 
# B = Tm['d02_0_00']-Tm['d03_0_00']
# dz = Z['d02']-Z['d03_op']
# lm = L['d02']


lr = (B/dz * 1000)



nc = Dataset('../data/wrf/geo_em.d03.nc')
lm = nc.variables['LANDMASK'][0,:,:]
glon,glat = hh.lonlat(nc)
ma = hh.map()

def landsea(r,d=5000):
	from pyproj import Geod
	from functools import partial
	inv = partial(Geod(ellps='WGS84').inv,r['lon'],r['lat'])
	def dist(x,y):
		return inv(x,y)[2]
	dv = np.vectorize(dist)
	return np.any(1-lm[np.where(dv(glon,glat)<d)])

l = sta.apply(landsea,1)
l = l[l].index

a = set(dz[dz<0].index) # -> station "a"bove grid level
b = set(dz[dz>0].index) # -> station "b"elow grid level
b_s = b.intersection(l)
a_s = a.intersection(l)
a_l,b_l,a_s,b_s,a,b = [list(s) for s in (a-a_s,b-b_s,a_s,b_s,a,b)]

# fig = plt.figure()
# x,y = d.loc[:,('lon','lat')].as_matrix().T
# ma.scatter(x,y,c=d[0],latlon=True)
# ma.drawcoastlines()
# plt.colorbar()
# fig.show()


def setup(ax,title=None,label=None):
	ax.grid(axis='x')
	ax.set_yticks([])
	ax.axvline(-9.8,color='r')
	ax.axvline(-6.5,color='g')
	if title is not None: ax.set_title(title)
	if label is not None: ax.set_ylabel(label,rotation=0,labelpad=20)
	
fig,ax = plt.subplots(2,2)
ax[0,0].hist(lr[b][lr.index.hour==0].stack(), range=(-20,20))
ax[0,0].hist(lr[b_s][lr.index.hour==0].stack(), range=(-20,20))
setup(ax[0,0],title='station < grid',label='0 h')

ax[0,1].hist(lr[a][lr.index.hour==0].stack(), range=(-20,20))
ax[0,1].hist(lr[a_s][lr.index.hour==0].stack(), range=(-20,20))
setup(ax[0,1],title='station > grid')

ax[1,0].hist(lr[b][lr.index.hour==12].stack(), range=(-20,20))
ax[1,0].hist(lr[b_s][lr.index.hour==12].stack(), range=(-20,20))
setup(ax[1,0],label='12 h')

ax[1,1].hist(lr[a][lr.index.hour==12].stack(), range=(-20,20))
ax[1,1].hist(lr[a_s][lr.index.hour==12].stack(), range=(-20,20))
setup(ax[1,1])
fig.show()

x = np.linspace(-20,20,100)
def kplot(y,label,bw=0.1):
	k = gaussian_kde(y.stack().dropna(),bw)
	m = brute(lambda z:-k(z),((-20,10),))
	plt.plot(x,k(x),label='{}: {:.2f}'.format(label,m[0]))
	return m[0]

def mode(y,bw=0.1):
	k = gaussian_kde(y.stack().dropna(),bw)
	m = brute(lambda z:-k(z),((-20,10),))
	return m[0]

def d02():
	fig = plt.figure() 

	plt.subplot(2,2,1)
	kplot(lr[b_l],'inland all')
	kplot(lr[b_l][lr.index.hour==0],'inland 0h')
	kplot(lr[b_l][lr.index.hour==12],'inland 12h')
	plt.axvline(-9.8,color='grey',ls='--')
	plt.axvline(-6.5,color='grey',ls=':')
	plt.grid()
	plt.legend()
	plt.title('station < grid')
	plt.gca().set_xlabel('T')

	plt.subplot(2,2,2)
	kplot(lr[a_l],'inland all',.001)
	kplot(lr[a_l][lr.index.hour==0],'inland 0h',.001)
	kplot(lr[a_l][lr.index.hour==12],'inland 12h',.001)
	kplot(lr[a_s],'coast all')
	kplot(lr[a_s][lr.index.hour==0],'coast 0h')
	kplot(lr[a_s][lr.index.hour==12],'coast 12h')
	plt.axvline(-9.8,color='grey',ls='--')
	plt.axvline(-6.5,color='grey',ls=':')
	plt.grid()
	plt.legend()
	plt.title('station > grid')
	plt.gca().set_xlabel('T')


	plt.subplot(2,2,3)
	plt.plot(lr[b].groupby(lr.index.hour).apply(mode), label='all')
	plt.plot(lr[b_l].groupby(lr.index.hour).apply(mode), label='land')
	plt.gca().set_xticks([0,6,12,18])
	plt.axhline(-9.8,color='grey',ls='--')
	plt.axhline(-6.5,color='grey',ls=':')
	plt.grid()
	plt.legend()
	plt.gca().set_xlabel('hour')
	plt.gca().set_ylabel('T')

	plt.subplot(2,2,4)
	plt.plot(lr[a].groupby(lr.index.hour).apply(lambda z:mode(z,.005)), label='all')
	plt.plot(lr[a_l].groupby(lr.index.hour).apply(lambda z:mode(z,.001)), label='land')
	plt.plot(lr[a_s].groupby(lr.index.hour).apply(mode), label='coast')
	plt.gca().set_xticks([0,6,12,18])
	plt.axhline(-9.8,color='grey',ls='--')
	plt.axhline(-6.5,color='grey',ls=':')
	plt.grid()
	plt.legend()
	plt.gca().set_xlabel('hour')
	plt.gca().set_ylabel('T')
	
fig = plt.figure() 

plt.subplot(2,2,1)
kplot(lr[b],'all')
kplot(lr[b_l],'inland all')
kplot(lr[b_l][lr.index.hour==0],'inland 0h')
kplot(lr[b_l][lr.index.hour==12],'inland 12h')
plt.axvline(-9.8,color='grey',ls='--')
plt.axvline(-6.5,color='grey',ls=':')
plt.grid()
plt.legend()
plt.title('station < grid')
plt.gca().set_xlabel('T')

plt.subplot(2,2,2)
kplot(lr[a],'all')
kplot(lr[a_s],'coast all')
kplot(lr[a_s][lr.index.hour==0],'coast 0h')
kplot(lr[a_s][lr.index.hour==12],'coast 12h')
plt.axvline(-9.8,color='grey',ls='--')
plt.axvline(-6.5,color='grey',ls=':')
plt.grid()
plt.legend()
plt.title('station > grid')
plt.gca().set_xlabel('T')


plt.subplot(2,2,3)
plt.plot(lr[b].groupby(lr.index.hour).apply(mode), label='all')
plt.plot(lr[b_l].groupby(lr.index.hour).apply(mode), label='land')
plt.gca().set_xticks([0,6,12,18])
plt.axhline(-9.8,color='grey',ls='--')
plt.axhline(-6.5,color='grey',ls=':')
plt.grid()
plt.legend()
plt.gca().set_xlabel('hour')
plt.gca().set_ylabel('T')

plt.subplot(2,2,4)
plt.plot(lr[a].groupby(lr.index.hour).apply(mode), label='all')
plt.plot(lr[a_s].groupby(lr.index.hour).apply(mode), label='coast')
plt.gca().set_xticks([0,6,12,18])
plt.axhline(-9.8,color='grey',ls='--')
plt.axhline(-6.5,color='grey',ls=':')
plt.grid()
plt.legend()
plt.gca().set_xlabel('hour')
plt.gca().set_ylabel('T')