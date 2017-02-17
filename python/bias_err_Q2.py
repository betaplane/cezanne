#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap
from matplotlib.figure import SubplotParams
import helpers as hh
import stipolate as st
import formulae as form
from functools import partial
from astropy.stats import LombScargle
import statsmodels.api as sm


K = 273.15
dd = lambda s: os.path.join('../data',s)

D = pd.HDFStore(dd('station_data.h5'))

sta = D['sta']
T = hh.extract(D['ta_c'],'prom') + K
RH = hh.extract(D['rh'],'prom')

S = pd.HDFStore(dd('LinearLinear.h5'))

def calc_rh(nc):
	T = nc.variables['T2'][:]
	qv = nc.variables['Q2'][:]
	p = nc.variables['PSFC'][:]
	rh = form.w2rh(qv,T,p)
	# apparently, this is necessary
	rh[rh<0] = 0
	rh[rh>100] = 100
	return rh

rh = S['RH2']
Z = S['z']
map = hh.map()

dz = [Z['d02']-sta['elev'],Z['d03_op']-sta['elev'],Z['d03_op']-sta['elev']]
def bias():
	fig = plt.figure(figsize=(10,6), subplotpars=SubplotParams(right=.88))
	for j,s in enumerate(['d02','d03_0_00','d03_0_12']):
		ax = plt.subplot(2,3,j+1)
		ax.set_title(s)
		b = (rh[s]-RH).mean()
		D = pd.concat((sta.loc[b.index,('lon','lat')], b),axis=1).dropna()
		m = 30
		sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-m, vmax=m))
		sm.set_array(D.iloc[:,-1])
		sm.set_cmap('coolwarm')
		for i,a in D.iterrows():
			map.plot(a['lon'],a['lat'],'o',color=sm.to_rgba(a.iloc[-1]),latlon=True)
		map.drawcoastlines()
		map.drawparallels(range(-32,-28,1),labels=[max(0,1-j),0,0,0])
		map.drawmeridians(range(-72,-69,1),labels=[0,0,0,1])
		
		if j==2:
			bb = ax.get_position()
			ax = fig.add_axes([bb.x1+0.02,bb.y0,0.02,bb.y1-bb.y0])
			plt.colorbar(sm, cax=ax)
			ax.set_title('$\Delta$RH', usetex=True)
		
		ax = plt.subplot(2,3,j+4)
		p = pd.concat((b,dz[j]),axis=1)
		plt.plot(p[0],p[1],'o')
		ax.set_xlabel('$\Delta$RH model-station', usetex=True)
		ax.grid()
		if j==1: ax.set_yticklabels([])
		if j==2: 
			ax.yaxis.set_ticks_position('right')
			ax.yaxis.set_label_position('right')
		if j!=1:
			ax.set_ylabel('$\Delta$z model-station', usetex=True)

def rms():
	fig = plt.figure(figsize=(10,6))
	for j,s in enumerate(['d02','d03_0_00','d03_0_12']):
		ax = plt.subplot(1,3,j+1)
		ax.set_title(s)
		b = (((rh[s]-RH)**2).mean()**.5).dropna()
	
		D = pd.concat((sta.loc[b.index,('lon','lat')], b),axis=1).dropna()
		sm = cm.ScalarMappable(norm=colors.Normalize(vmin=8, vmax=40))
		sm.set_array(D.iloc[:,-1])
		sm.set_cmap('gnuplot')
		for i,a in D.iterrows():
			map.plot(a['lon'],a['lat'],'o',color=sm.to_rgba(a.iloc[-1]),latlon=True)
		map.drawcoastlines()
		map.drawparallels(range(-32,-28,1),labels=[max(1-j,0),0,0,0])
		map.drawmeridians(range(-72,-69,1),labels=[0,0,0,1])
		if j==2:
			bb = ax.get_position()
			ax = fig.add_axes([bb.x1+0.02,bb.y0,0.02,bb.y1-bb.y0])
			plt.colorbar(sm, cax=ax)

def pow(T,d):
	try:
		c = d.dropna()
		t = np.array(c.index,dtype='datetime64[h]').astype(float)
		if max(t)-min(t)<T/4: return np.nan
		x = c.as_matrix()
		y = LombScargle(t,x).model(np.linspace(0,T,100),1/T)
		return max(y)-min(y)
	except: return np.nan


# nc = Dataset(dd('wrf/d03_day0.nc'))


# get the overlap between the 00-started and 12-started operational forecasts
# i = np.where(np.diff(t).astype(int)<=0)[0][0]+1

def cycles():
	B = dict(rh.iteritems())
	B.update({'obs':RH})
	B = pd.Panel(B)
	fig = plt.figure(figsize=(10,6),subplotpars=SubplotParams(left=.08))
	for k in range(2):
		sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax={0:30,1:60}[k]))
		for j,s in enumerate(['obs','d02','d03_0_00','d03_0_12']):
			ax = plt.subplot(2,4,4*k+j+1)
			if k==0: ax.set_title(s)
			b = B[s].apply(partial(pow,{0:24*365.25,1:24}[k])).dropna()
			D = pd.concat((sta.loc[b.index,('lon','lat')], b),axis=1).dropna()	
			sm.set_array(D.iloc[:,-1])
			sm.set_cmap('gnuplot')
			for i,a in D.iterrows():
				map.plot(a['lon'],a['lat'],'o',color=sm.to_rgba(a.iloc[-1]),latlon=True)
			map.drawcoastlines()
			map.drawparallels(range(-32,-28,1),labels=[max(0,1-j),0,0,0])
			map.drawmeridians(range(-72,-69,1),labels=[0,0,0,k])
			if j==3:
				bb = ax.get_position()
				ax = fig.add_axes([bb.x1+0.02,bb.y0,0.02,bb.y1-bb.y0])
				plt.colorbar(sm, cax=ax)

Tm = S['T2']
P = pd.HDFStore('../data/pressure.h5')
a,b = P['fit']
p = np.exp(a + sta['elev']*b)
w = form.rh2w(RH,T,p*100)
Q2 = S['Q2']

# scatterplots and linear regression bias-RH
def all():
	"raw data"
	r2 = np.zeros((3,5))
	c = np.zeros((3,5))
# 	fig, axs = plt.subplots(3,5)
	def scatter(x,y,i,j):
		global r2,c,ax
		m = pd.concat((y,x),axis=1).dropna(0,'any').as_matrix()
		lm = sm.OLS(m[:,0],sm.add_constant(m[:,1]))
		r = lm.fit()
		r2[j,i] = r.rsquared_adj
		c[j,i] = r.params[1]
# 		axs[j,i].plot(m[:,1],m[:,0],'+')	

	for i,s in enumerate(['d02','d03_0_00','d03_0_12','d03_orl','fnl']):
		dt = (Tm[s]-T).stack()
		scatter(RH.stack(),dt,i,0)	
		if i<3:
			dr = rh[s]-RH
			scatter(dr.stack(),dt,i,1)
			scatter(rh[s].stack(),dt,i,2)

	
def part():
	"daily & monthly averages"
	r2 = np.zeros((3,5))
	c = np.zeros((3,5))
	fig, axs = plt.subplots(3,5)
	def scatter(x,y,i,j):
		global r2,c,ax
		m = pd.concat((y,x),axis=1).dropna(0,'any')
		m = m.unstack()
		m = m.groupby(m.index.date).mean().stack().as_matrix()
# 		m = m.groupby((m.index.year,m.index.month)).mean().stack().as_matrix()
# 		m = m.mean().unstack().as_matrix().T
		lm = sm.OLS(m[:,0],sm.add_constant(m[:,1]))
		r = lm.fit()
		r2[j,i] = r.rsquared_adj
		c[j,i] = r.params[1]
		axs[j,i].plot(m[:,1],m[:,0],'+')

	for i,s in enumerate(['d02','d03_0_00','d03_0_12','d03_orl','fnl']):
		dt = (Tm[s]-T).stack()
		scatter(w.stack(),dt,i,0)
		axs[0,i].set_title(s)
		if i<3:
			dr = rh[s]-w
			scatter(dr.stack(),dt,i,1)
			scatter(rh[s].stack(),dt,i,2)

def qq():
	r2 = np.zeros((3,3))
	c = np.zeros((3,3))
	fig, axs = plt.subplots(3,3)
	def scatter(x,y,i,j):
		global r2,c,ax
		m = pd.concat((y,x),axis=1).dropna(0,'any')
		m = m.unstack()
# 		m = m.groupby(m.index.date).mean().stack().as_matrix()
# 		m = m.groupby((m.index.year,m.index.month)).mean().stack().as_matrix()
		m = m.mean().unstack().as_matrix().T
		lm = sm.OLS(m[:,0],sm.add_constant(m[:,1]))
		r = lm.fit()
		r2[j,i] = r.rsquared_adj
		c[j,i] = r.params[1]
		axs[j,i].plot(m[:,1],m[:,0],'+')

	for i,s in enumerate(['d02','d03_0_00','d03_0_12']):
		dt = Tm[s]-T
# 		dt -= dt.mean()
		dt = (dt+0.0065*(Z[s if s=='d02' else 'd03_op']-sta['elev']))
		dt = dt.stack()
		scatter(w.stack(),dt,i,0)
# 		scatter((w-w.mean()).stack(),dt,i,0)
		axs[0,i].set_title(s)
		if i<3:
			dr = Q2[s]-w
# 			dr -= dr.mean()
			scatter(dr.stack(),dt,i,1)
			scatter(Q2[s].stack(),dt,i,2)
# 			scatter((Q2[s]-Q2[s].mean()).stack(),dt,i,2)

