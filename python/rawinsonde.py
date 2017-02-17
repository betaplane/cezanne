#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
from astropy.stats import LombScargle
from glob import glob
from netCDF4 import Dataset
from datetime import datetime
import helpers as hh
import os


K = 273.15
dd = lambda s: os.path.join('../data',s)

def read(f):
	d = pd.read_csv(f,header=None,delim_whitespace=True)
	t = [pd.Period('{}-{}'.format(r[1],r[2]),freq='M') for i,r in d.iterrows()]
	d.index = pd.MultiIndex.from_arrays((t,d[3]))
	d = d.drop([0,1,2,3],1)
	d.columns = ['value','num']
	return d


def key(f):
	s = os.path.splitext(os.path.split(f)[1])[0].split('_')
	return '{}{}'.format(s[2][0],s[1][:2])

T = pd.Panel(dict([(key(f),read(f)) for f in glob(dd('IGRA/temp_*_*'))]))
# Tm = T.minor_xs('value').apply(lambda a:a.unstack().mean())

# G = pd.Panel(dict([(key(f),read(f)) for f in glob(dd('IGRA/ghgt_*_*'))]))
# Gm = G.minor_xs('value').apply(lambda a:a.unstack().mean())

# p = pd.Panel({'g':G.minor_xs('value'), 'T':T.minor_xs('value')}).transpose(2,1,0)


# wrf files
g = Dataset(dd('wrf/geo_em.d02.nc'))
lon = g.variables['XLONG_M'][:].squeeze()
lat = g.variables['XLAT_M'][:].squeeze()

nc = Dataset('d02_2014-09-10_p_intp.nc')
t = nc.variables['temp'][:]
lvl = nc.variables['level'][:]/100
time = hh.get_time(nc).astype(datetime)

# use only one year of data to avoid seasonal bias
k = np.where(time==datetime(2014,12,31,00))[0][0]
t = t[k:,:,:,:]
time = time[k:]


# FNL (converted to netcdf)
# nc = Dataset('../data/fnl/T.nc2')
# t = nc.variables['t'][:]
# lvl = nc.variables['lev'][:]/100
# time = hh.get_time(nc).astype(datetime)
# lon = nc.variables['lon'][:]-360
# lat = nc.variables['lat'][:]
# i = np.where((lon>-74)&(lon<-68))[0]
# j = np.where((lat>-33)&(lat<-27))[0]
# k = np.where((time>=datetime(2014,12,31))&(time<datetime(2015,12,31)))[0]
# t = t[k[0]:k[-1]+1,:,j[0]:j[-1]+1,i[0]:i[-1]+1]
# lon,lat = np.meshgrid(lon[i],lat[j])
# time = time[k]

# lon,lat = np.meshgrid(lon,lat)


# wrf UPP grib
# from grib import ungrib
# t, time, lvl, lat, lon = ungrib('t_upp_2015.grb1')
# time = time.astype(datetime)


def profile():
	# model mean
	ts = pd.DataFrame(np.nanmean(t.reshape((t.shape[0], t.shape[1], -1)),2), index=time, columns=lvl)

	# std in space of time mean
	t_std_space = np.nanstd(np.nanmean(t,0).reshape((lvl.shape[0],-1)),1)

	# for comparison with data, only compute std of monthly means
	t_std_time = ts.groupby(ts.index.month).mean().std()

	# extract the times and places closest to radiosonde data
	# average over both daily soundings, since they are very similar
	k = np.where([(i.hour==0 or i.hour==12) for i in time])[0]

	# Sto. Domingo
	i,j,d = hh.nearest(lon,lat,-71.6144,-33.6547)
	t_sd = pd.DataFrame(t[k,:,i,j], index=time[k], columns=lvl)
	t_sd = t_sd.groupby(t_sd.index.month).mean()
	T_sd = T.xs(['s00','s12'],'items').xs('value','minor').mean(1).unstack()
	# perform mean by month first, to avoid seasonal bias
	T_sd = T_sd.groupby(T_sd.index.month).mean().drop(9999,1) / 10 + K

	# Mendoza
	i,j,d = hh.nearest(lon,lat,-68.7833,-32.8333)
	t_men = pd.DataFrame(t[k,:,i,j], index=time[k], columns=lvl)
	t_men = t_men.groupby(t_men.index.month).mean()
	T_men = T.xs(['m00','m12'],'items').xs('value','minor').mean(1).unstack()
	# perform mean by month first, to avoid seasonal bias
	T_men = T_men.groupby(T_men.index.month).mean().drop(9999,1) / 10 + K

	fig = plt.figure()
	plt.errorbar(T_sd.mean(), T_sd.columns, xerr=T_sd.std(), capsize=3, label='Sto Domingo')
	plt.errorbar(T_men.mean(), T_sd.columns, xerr=T_men.std(), capsize=3, label='Mendoza')
	plt.errorbar(t_sd.mean(), lvl, xerr=t_sd.std(), capsize=3, label='Sto Domingo model')
	plt.errorbar(t_men.mean(), lvl, xerr=t_men.std(), capsize=3, label='Mendoza model')
	plt.errorbar(ts.mean(), lvl, xerr=t_std_space, capsize=3, label='model avg')

	ax = fig.axes[0]
	ax.invert_yaxis()
	ax.set_xlabel('T [K]')
	ax.set_ylabel('p [hPa]')
	ax.set_yscale('log')
	ax.set_ylim([1030,60])
	plt.legend()
	fig.show()



def model(d):
	i = d.asfreq('1H')
	d = d.dropna()
	nt = lambda d:np.array(d.index,dtype='datetime64[h]').astype(float)
	return LombScargle(nt(d),d).model(nt(i),1/24)

p = {}
for s in (('St. Domingo',-71.6144,-33.6547),('Mendoza',-68.7833,-32.8333)):
	i,j,d = hh.nearest(lon,lat,s[1],s[2])
	x = pd.DataFrame(t[:,:,i,j], index=time, columns=lvl)
	z = x.iloc[:,x.any().values].apply(model,0)
	y = pd.concat((
		T[s[0][0].lower()+'00']['value'].unstack().mean(),
		T[s[0][0].lower()+'12']['value'].unstack().mean()
	), axis=1)
	y.columns = [0,12]
# 	m = pd.DataFrame(np.nanmean(t.reshape((t.shape[0],t.shape[1],-1)),2), index=time, columns=lvl)
	p.update({
		(s[0],'mod'): x.groupby(x.index.hour).mean(),
		(s[0],'obs'): y.drop(9999).transpose()/10 + K,
		(s[0],'sin'): z.groupby(z.index.hour).mean(),
# 		(s[0],'ave'): (m-m.mean()).groupby(m.index.hour).mean()
	})
P = pd.Panel(p)


rows = 10
fig,axs = plt.subplots(rows,2,subplotpars=SubplotParams(right=.88,top=.95,bottom=.05))
lvls = T.major_axis.levels[1].drop(9999)[-rows:]
for j in range(rows):
	l = lvls[j]
	for k,s in enumerate(P.items.levels[0]):
		if ~P[(s,'obs')][l].any():
			axs[j,k].set_axis_off()
			continue
		axs[j,k].plot(P.major_axis,P[(s,'mod')][l])
		axs[j,k].plot(P.major_axis,P[(s,'obs')][l],'x')
# 		ax.plot(P.major_axis,P[s+'ave'][l])
		axs[j,k].plot(P.major_axis,P[(s,'sin')][l])
		axs[j,k].set_xticks([0,6,12,18])
		axs[j,k].set_xlim([-1,24])
		axs[j,k].yaxis.get_major_formatter().set_useOffset(False)
		if j<rows-1 and P[(s,'obs')][lvls[j+1]].any(): axs[j,k].set_xticklabels([])
		if j==0: axs[j,k].set_title(s)
		if k==1: axs[j,k].yaxis.set_ticks_position('right')
	b1 = axs[j,0].get_position()
	b2 = axs[j,1].get_position()
	ax = fig.add_axes([b1.x1,b1.y0,b2.x0-b1.x1,b1.y1-b1.y0])
	ax.set_axis_off()
	ax.text(.5,.5,l,ha='center',va='center')
		