#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os, pygrib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.gridspec as gs
import stipolate as st
import scipolate as sp
import mapping as mp
import helpers as hh
from scipy import interpolate as ip
from functools import partial


K = 273.15
dd = lambda s: os.path.join('../data',s)

D = pd.HDFStore(dd('station_data.h5'))

sta = D['sta']
T = hh.extract(D['ta_c'],'prom') + K



def interp2D(method='nearest', zmethod='nearest'):
	T0 = st.interp_grib(pygrib.open('data/fnl/T2.grib2'), sta, method=method)
	T0.sort_index(inplace=True)
	
	z,lat,lon = pygrib.open('data/fnl/oro.grb2')[1].data()
	lon -= 360
	# cut off +/- 90 latitude
	z0 = sp.grid_interp((lon[1:-1,:],lat[1:-1,:]), z[1:-1,:], hh.lonlat(sta), sta.index, method=zmethod)
 	
	print('d01')
	g1 = Dataset('data/wrf/d01_T2_2014-09-10.nc')
	T1 = st.interp_nc(g1,'T2',sta, method=method)
	z1 = st.interp_nc(g1,'HGT', sta, time=False, method=zmethod)

	print('d02')
	g2 = Dataset('data/wrf/d02_2014-09-10.nc')
	T2 = st.interp_nc(g2,'T2',sta, method=method)
	z2 = st.interp_nc(g2,'HGT', sta, time=False, method=zmethod)

	print('d03')
	g3 = Dataset('data/orlando/T2.nc')
	T3 = st.interp_nc(g3,'T2',sta, method=method)
	z3 = st.interp_nc(g3,'HGT', sta, time=False, method=zmethod)
	
	Z = pd.DataFrame({'d01':z1,'d02':z2,'d03':z3,'fnl':z0})
	V = pd.Panel.from_dict({'d01':T1,'d02':T2,'d03':T3,'fnl':T0}, orient='minor')
	return V,Z
	
def prepare(T,Tm,Zm):
	d = {}
	for k in ['fnl','d01','d02','d03']:
		t = Tm.minor_xs(k)
		z = Zm[k]
		bias = (t-T).mean()
		dz = z-sta['elev']
		d[k] = pd.concat({
			'lon': sta['lon'],
			'lat': sta['lat'],
			'elev': sta['elev'],
			'hgt': z, 
			'dz': dz, 
			'bias': bias,
			'lapse': bias+6.5/1000*dz
		},axis=1)
	return pd.Panel.from_dict(d, orient='minor')

def make_fig(D):
	a = list(zip(['fnl','d01','d02','d03'],['1','+','x','*']))
	
	fig = plt.figure()
	ax = plt.subplot(2,2,1)
	for k,m in a:
		ax.scatter(D['bias'][k],D['dz'][k],marker=m,label=k)
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.set_xlabel('bias')
	ax.set_ylabel('elevation grid-station')
	ax.legend()
	ax.grid()

	ax = plt.subplot(2,2,2)
	for k,m in a:
		ax.scatter(D['lapse'][k],D['dz'][k],marker=m,label=k)
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position('right')
	ax.set_ylabel('elevation grid-station')
	ax.set_xlabel('bias + 6.5K/1000m')
	ax.legend(loc=4)
	ax.grid()

	ax = plt.subplot(2,2,3)
	for k,m in a:
		ax.scatter(D['bias'][k],D['elev'][k],marker=m,label=k)
	ax.set_xlabel('bias')
	ax.set_ylabel('station elevation')
	ax.legend(loc=2)
	ax.grid()

	ax = plt.subplot(2,2,4)
	for k,m in a:
		ax.scatter(D['bias'][k],D['hgt'][k],marker=m,label=k)
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position('right')
	ax.set_xlabel('bias')
	ax.set_ylabel('grid elevation')
	ax.legend()
	fig.show()
	ax.grid()


def lsq(Y, X, b0=True):
	if b0:
		b = np.linalg.lstsq(np.r_['1,2',np.ones((X.shape[0],1)),X],Y)[0]
	else:
		b = np.linalg.lstsq(X,Y)[0]
	return dict(zip(['b{}'.format(i) for i in np.arange(b.shape[0])+(1-b0)], b))
	
def lsq1(X, Y, b0=True):
	y = Y.stack()
	x = np.reshape([X[i[1]] for i in y.index], (len(y),1))
	return lsq(y, x, b0=b0)
	
def lsq2(z,dz,b,b0=True):
	Y = b.stack()
	X = np.array([(z[i[1]],dz[i[1]]) for i in Y.index])
	return lsq(Y, X, b0=b0)
	
def pca(*x):
	from sklearn.decomposition import PCA
	X = pd.concat(x, axis=1).dropna().as_matrix()
	p = PCA(X.shape[1])
	p.fit(X)
	return p.transform(X)


S = pd.HDFStore(dd('LinearLinear.h5'))
Tm = S['T2']
Z = S['z']
r = S['lapse']


def lsq_table(T,Tm,Z,b0=True,sign=None):
	"""
Makes a table with lapse rates determining by least-squares regressing the bias (model minus
stations) onto dz (model grid elevation minus station elevation).
'all': fit performed using all bias data points
'mean': fit performed only temporal means for each station
'zdz': multiple regression on dz and z (the grid cell elevation)
	"""
	d = {}
	for x in Tm.minor_axis:
		z = Z[x]
		dz = z - sta['elev']
		s = dz.index if sign is None else dz[np.sign(dz)==sign].index
		b = Tm[s].minor_xs(x) - T
		d[x] = {
			'all': lsq1(dz[s], b, b0=b0), 
			'mean': dict(zip(('b0','b1'), hh.lsq(dz[s], b.mean(), b0=b0))),
			'zdz': lsq2(z, dz, b, b0=b0)
		}
	return pd.Panel(d).transpose(0,2,1).to_frame()

# lapse = dict([(i,lsq_table(T,Tm,Z,sign=s)) for i,s in zip(('all','dz>0','dz<0'),(None,1,-1))])
# lapse = pd.Panel(lapse).transpose(2,0,1).to_frame()
# lapse.index.names = ['dz','method','beta']
# S['lapse_rates'] = lapse


def common_dz(T,Tm,Z,b0=True,sign=None):
	"""
Performs least-squares regression across several different data sets, fitting a single slope
but allowing for different offsets (i.e., mean biases).
	"""
	Y = []
	for a in Tm.minor_axis:
		z = Z[a]
		dz = z - sta['elev']
		s = dz.index if sign is None else dz[np.sign(dz)==sign].index
		b = (Tm[s].minor_xs(a) - T).stack()
		x = np.reshape([dz[i[1]] for i in b.index], (len(b),1))
		Y = np.r_[Y, b]
		try: 
			X = np.pad(X,((0,0),(1,0)),mode='constant')
			x = np.pad(x,((0,0),(X.shape[1]-2,0)),mode='constant')
			x = np.r_['1,2',np.ones((x.shape[0],1)),x]
		except NameError: 
			X = np.r_['1,2',np.ones((x.shape[0],1)),x]
		else:
			X = np.r_[X,x]
	return np.linalg.lstsq(X,Y)[0]
		
# lapse = pd.Series([common_dz(T,Tm,Z,sign=s)[-1] for s in [None,1,-1]], index=('all','dz>0','dz<0'))
# S['lapse'] = lapse

def subhist(grid,i,splot,lr):
	Tm1 = pd.concat((Th,Tl1),axis=1)
	Tm2 = pd.concat((Th,Tl2),axis=1)

	ax = plt.subplot(grid[1:4,i*3+1:])
	c = (Tm0-T).stack()
	splot(ax,c)
	ax.set_title('d02 other')
	
	ax = plt.subplot(grid[4:7,i*3+1:])
	dzh = dz[dz>0]
	Tm = pd.concat((Tm0[dzh.index]-dzh*lr, Th), axis=1)
	c = (Tm-T).stack()
	splot(ax,c)
	
	ax = plt.subplot(grid[7:10,i*3+1:])
	c = (Tm1-T).stack()
	splot(ax,c)
	
	ax = plt.subplot(grid[10:13,i*3+1:])
	c = (Tm2-T).stack()
	splot(ax,c,True)
	ax.set_xlabel('K')
	

def histo(r):
	fig = plt.figure()
	I = len(Tm.minor_axis) + 1
	xlim = [(-20,20),(-20,20),(-20,20),(-25,25)]
	xticks = np.repeat([[-20,-10,0,10,20]],4,0)
	ylim = [(0,65000),(0,225000),(0,350000),(0,150000)]
	yticklabels = [np.array([0,2,4,6]),np.array([0,5,10,15,20]),np.array([0,1,2,3]),np.array([0,5,10,15])]
	ylabel = ['10k','10k','100k','10k']
	grid = gs.GridSpec(13,16)
	grid.update(left=.06, right=.98, wspace=2)
	
	def splot(i, ax, c=None, xticklabels=False):
		yticks = yticklabels[i] * int(ylabel[i].replace('k','000'))
		ax.set_ylim(ylim[i])
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels[i])
		ax.set_ylabel(ylabel[i])
		ax.grid()
		if c is not None:
			ax.hist(c)
			ax.set_xlim(xlim[i])
			ax.set_xticks(xticks[i])
			ax.text(.05,.85, np.sum(c**2)**.5/len(c), transform=ax.transAxes)
		if xticklabels==False:
			ax.set_xticklabels([])
		
		
	for i,a in enumerate(Tm.minor_axis):
		z = Z[a]
		dz = z - sta['elev']
		b = Tm.minor_xs(a) - T
		v,w = pd.concat((dz, b.notnull().sum()), axis=1).dropna().as_matrix().astype(int).T
		ax = plt.subplot(grid[0:3,i*3:(i+1)*3])
		ax.hist(v, weights=w)
		ax.invert_xaxis()
		splot(i,ax,xticklabels=True)
		ax.set_title(a)
		ax.set_xlabel('m')
		
		ax = plt.subplot(grid[4:7,i*3:(i+1)*3])
		c = b.stack()
		splot(i,ax,c)
		
		ax = plt.subplot(grid[7:10,i*3:(i+1)*3])
		c = (b-dz*r[0]).stack()
		splot(i,ax,c)
		
		ax = plt.subplot(grid[10:13,i*3:(i+1)*3])
		c = np.r_[
			(b[dz[dz>0].index]-dz*r[1]).stack(), 
			(b[dz[dz<0].index]-dz*r[2]).stack()
		]
		splot(i,ax,c,True)
		ax.set_xlabel('K')
	subhist(grid,4,partial(splot,1),r[0])
	fig.show()


	
def interp4D():
	nc = Dataset('data/wrf/d02_2014-09-10.nc')
	m = mp.basemap(nc)
	xy = m(*hh.lonlat(nc))
	ij = m(*hh.lonlat(sta))
	t = hh.get_time(nc) - np.timedelta64(4,'h')

	z = sp.grid_interp(xy,nc.variables['HGT'][:],ij, sta.index, method='linear')
	TH2 = sp.grid_interp(xy,nc.variables['TH2'][:],ij, sta.index, t, method='linear')
	TSK = sp.grid_interp(xy, nc.variables['TSK'][:], ij, sta.index, t, method='linear')
	GP = sp.interp4D(xy,nc.variables['PH'][:],ij,sta.index,t,'linear')
	GPB = sp.interp4D(xy,nc.variables['PHB'][:],ij,sta.index,t,'linear')
	P = sp.interp4D(xy,nc.variables['P'][:],ij,sta.index,t,'linear')
	PB = sp.interp4D(xy,nc.variables['PB'][:],ij,sta.index,t,'linear')
	TH = sp.interp4D(xy,nc.variables['T'][:],ij,sta.index,t,'linear')

	V = pd.HDFStore('data/d02_4D.h5')
	V['GP'] = GPB.add(GP)
	V['P'] = PB.add(P)
	V['T'] = TH
	V['TH2'] = TH2
	V['TSK'] = TSK
	V['z'] = z


V = pd.HDFStore(dd('d02_4D.h5'))
z = V['z']
dz = z - sta['elev']
TH = V['T'] + 300
TSK = V['TSK']
TH2 = V['TH2']
P = V['P']
GP = V['GP']/9.8

PS = pd.HDFStore(dd('pressure.h5'))
b = PS['fit']
p = np.exp(b['b0'] + b['b1']*z)
T2 = TH2*(p/1000)**.286

# c = (T2-T).stack()
# fig = plt.figure()
# plt.hist(c)
# fig.show()

def higher(index):
	T = {}
	for i in index:
		print(i)	
		T[i] = {}
		for t,r in GP[i].iterrows():
			z = r.as_matrix()
			z = (z[:-1]+z[1:]) / 2
			th = ip.interp1d(z, TH[i].loc[t], kind='linear')(sta['elev'][i]+40)
			p = np.exp(ip.interp1d(z, np.log(P[i].loc[t]), kind='linear')(sta['elev'][i]+40))
			T[i][t] = (th * (p/100000) ** .286 + TSK[i][t]) / 2
	return pd.DataFrame(T)
	
def lower1(index):
	T = {}
	for i in index:	
		print(i)
		T[i] = {}
		for t,r in GP[i].iterrows():
			z = np.sum(r[:2]) / 2
			temp = TH[i].loc[t][0] * (P[i].loc[t][0]/100000) ** .286
			T[i][t] = (temp + 6.5 * (z-sta.loc[i]['elev']-40)/1000 + TSK[i][t]) / 2
	return pd.DataFrame(T)
	
def lower2(index, b0, b1):
	T = {}
	for i in index:		
		print(i)
		T[i] = {}
		for t,r in TH[i].iterrows():
			p = np.exp(b0 + b1*(sta.loc[i]['elev']+40))
			T[i][t] = (r[0] * (p/1000) ** .286 + TSK[i][t]) / 2
	return pd.DataFrame(T)		

			


Th = higher(dz[dz<0].index)
Tm0 = Tm.minor_xs('d02')
Th = V['T2_higher']
Tl1 = V['T2_lower_lr']
Tl2 = V['T2_lower_th']
# 
# histo2(Tm0, Th, Tl1, Tl2, dz)