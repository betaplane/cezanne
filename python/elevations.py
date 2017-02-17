#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import stipolate as st
from netCDF4 import Dataset
import helpers as hh

K = 273.15
dd = lambda s: os.path.join('../data',s)


def elev_correct(Tm,Z,lr):
	def correct(x):
		dz = Z[x] - sta['elev']
		return Tm.minor_xs(x) - lr*dz
	return pd.Panel.from_dict(dict([(x,correct(x)) for x in Tm.minor_axis]),orient='minor')

def binned_plot1(self, x, values, color=None, label=None):
	me,b,n = binned_statistic(x,values,'mean',50)
	mi = binned_statistic(x,values,np.nanmin,50)[0]
	ma = binned_statistic(x,values,np.nanmax,50)[0]
	xc = (b[:-1]+b[1:])/2
	self.fill_betweenx(xc, mi, ma, color=color, alpha=.4)
	self.plot(me, xc, color=color, label=label)

def binned_plot2(self, x, values, color=None, label=None):
	me,b,n = binned_statistic(x,values,'mean',50)
	std = binned_statistic(x,values,np.nanstd,50)[0]
	xc = (b[:-1]+b[1:])/2
	self.fill_betweenx(xc, me-2*std, me+2*std, color=color, alpha=.4)
	self.plot(me, xc, color=color, label=label)

Axes.binned_plot = binned_plot2

# model, complete field
nc = Dataset(dd('wrf/d02_2014-09-10.nc'))
z = nc.variables['HGT'][:].flatten()

# use only one year of data so as to not bias towards a particular season
j = np.where(hh.get_time(nc)==np.datetime64('2014-12-31T00'))[0][0]
T2 = nc.variables['T2'][:]
T2 = T2[j:,:,:]
T2 = np.mean(T2,0).flatten()
nc.close()


# model, field interpolated to station locations
S = pd.HDFStore(dd('LinearLinear.h5'))
Tm = S['T2'].minor_xs('d02').dropna(0,'all')

# use only one year of data, s.a.
Tm = Tm[Tm.index>='2014-12-31'].mean()
Z = S['z']
# lr = S['lapse']['all']
lr = -6.5/1000


# station data
D = pd.HDFStore(dd('station_data.h5'))
sta = D['sta']
T = hh.extract(D['ta_c'],'prom') + K

# first, average over the same day in different years to avoid bias towards those that occur more often
T = T.groupby(T.index.dayofyear).mean().mean()

a = pd.concat((T,sta['elev']),axis=1,keys=['T','z']).sort_values('z')
b = pd.concat((Tm,Z['d02']),axis=1,keys=['T','z']).sort_values('z')
c = pd.concat((Tm-lr*(Z['d02']-sta['elev']),sta['elev']),axis=1,keys=['T','z']).sort_values('z')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1)
ax.binned_plot(z,T2,color=colors[0], label='model mean / elev')
ax.scatter(b['T'],b['z'], marker='+', color=colors[1], label='model station loc no adj')
ax.scatter(c['T'],c['z'], marker='+', color=colors[2], label='model station loc adj for {} K/km'.format(round(lr*1000,2)))
ax.scatter(a['T'],a['z'], marker='+', color=colors[3], label='observations')
ax.legend(loc=3)
ax.set_xlabel('T [K]')
ax.set_ylabel('elev [m]')
fig.show()

# nc = Dataset(dd('wrf/d02_2014-09-10_transf.nc'))
# GP = nc.variables['ghgt'][:]
# gpm = np.mean(GP,0).reshape((GP.shape[1],-1))
# temp = nc.variables['temp'][:]
# temm = np.mean(temp,0).reshape((temp.shape[1],-1))
# P = nc.variables['p'][:]

