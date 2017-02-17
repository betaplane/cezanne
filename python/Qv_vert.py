#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime
import helpers as hh


d = lambda x:x.transpose([2,0,1])['value'].drop(9999).transpose([1,2,0])

S = pd.HDFStore('../data/IGRA/IGRAmly.h5')
v = d(S['vapr'])/10

T = d(S['temp'])/10
es = 610.94 * np.exp((17.625*T).div(T+243.04))
# w = 0.622 * es.div(np.r_['1,3',T.minor_axis*100] - es)
rh = 100 * v.div(es)

g = Dataset('../data/wrf/geo_em.d02.nc')
lon = g.variables['XLONG_M'][:].squeeze()
lat = g.variables['XLAT_M'][:].squeeze()

# nc = Dataset('d02_e_pint_lin.nc')
# e = nc.variables['e'][:]
# p = nc.variables['level'][:]
# time = hh.get_time(nc).astype(datetime)
# # use only one year of data to avoid seasonal bias
# k = np.where(time==datetime(2014,12,31,00))[0][0]
# time = time[k:]
# e = e[k:,:,:,:]
# em = np.nanmean(e.transpose([1,0,2,3]).reshape((len(p),-1)),1)

nc = Dataset('d02_rh_pint_lin.nc')
r = nc.variables['rh'][:]
p = nc.variables['level'][:]
time = hh.get_time(nc).astype(datetime)
# use only one year of data to avoid seasonal bias
k = np.where(time==datetime(2014,12,31,00))[0][0]
time = time[k:]
r = r[k:,:,:,:]
rm = np.nanmean(r.transpose([1,0,2,3]).reshape((len(p),-1)),1)
nc.close()

coords = {'stdomingo':(-71.6144,-33.6547),'mendoza':(-68.7833,-32.8333)}
fig,ax = plt.subplots()
for x in ['stdomingo','mendoza']:
	for h in v[x].items:
# 		q = v[x][h].mean().dropna()
		q = rh[x][h].mean().dropna()
		pl = plt.plot(q,q.index,'-s',label='{} {:02d}h obs'.format(x,h))[0]
		i,j,d = hh.nearest(lon,lat,*coords[x])
		k = np.where([t.hour==h for t in time])[0]
# 		s = e[k,:,i,j]
		s = r[k,:,i,j]
		plt.plot(np.nanmean(s,0),p/100, '-^', color=pl.get_color(), label='{} {:02d}h mod'.format(x,h))

# plt.plot(em,p/100,'-o',label='d02 ave')
plt.plot(rm,p/100,'-o',label='d02 ave')
plt.legend()
ax.invert_yaxis()
ax.set_ylim([1020,400])
ax.set_xlabel('vapor pressure [Pa]')
ax.set_xlabel('RH [%]')
ax.set_ylabel('pressure [hPa]')
ax.set_title('d02')
plt.grid()
fig.show()


nc = Dataset('../data/wrf/d02_2014-09-10_transf.nc')
r = nc.variables['rh'][:]
gh = nc.variables['ghgt'][:]
time = hh.get_time(nc).astype(datetime)
# use only one year of data to avoid seasonal bias
k = np.where(time==datetime(2014,12,31,00))[0][0]
time = time[k:]
r = r[k:,:,:,:]
rm = np.nanmean(r.transpose([1,0,2,3]).reshape((gh.shape[1],-1)),1)
gm = np.nanmean(gh.transpose([1,0,2,3]).reshape((gh.shape[1],-1)),1)
nc.close()

g = d(S['ghgt'])

D = pd.HDFStore('../data/station_data.h5')
sta = D['sta']
RH = hh.extract(D['rh'],'prom')
b = pd.concat((RH.mean(),sta['elev']),axis=1)

fig,ax = plt.subplots()
for x in ['stdomingo','mendoza']:
	for h in rh[x].items:
		q = rh[x][h].mean().dropna()
		z = g[x][h].mean()
		pl = plt.plot(q,z[q.index],'-s',label='{} {:02d}h obs'.format(x,h))[0]
		i,j,d = hh.nearest(lon,lat,*coords[x])
		k = np.where([t.hour==h for t in time])[0]
		plt.plot(
			np.nanmean(r[k,:,i,j],0),
			np.nanmean(gh[k,:,i,j],0),
			'-^', color=pl.get_color(), label='{} {:02d}h mod'.format(x,h)
		)
plt.plot(rm,gm,'-D',label='d02 ave')		
plt.plot(b[0],b['elev'],'o',label='stations')
ax.set_xlabel('RH [%]')
ax.set_ylabel('geopot [m]')
ax.set_ylim([0,10000])
plt.legend()
