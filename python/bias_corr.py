#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from netCDF4 import Dataset
import helpers as hh
from scipy.stats import gaussian_kde
import re


D = pd.HDFStore('../../data/tables/station_data.h5')

sta = D['sta']
T = hh.extract(D['ta_c'],'prom',True)

S = pd.HDFStore('../../data/tables/LinearLinear.h5')
Tm = S['T2']
Z = S['z']

x = np.linspace(-20,20,100)
z = re.compile('d03_\d')


def abcd(df):
	m = df.notnull().astype(int)
	n = df.shape[0]
	a = np.where(m.sum(axis=1)==2)[0].shape[0]
	b,c = [np.where(m.diff(axis=1)[1]==i)[0].shape[0] for i in [1,-1]]
	d = n-a-b-c
	return a,b,c,d,n


cols = np.array([['fnl','d01','d02'],['d03_orl','d03_0_00','d03_0_12'],['d03_0_00','d03_1_00','d03_4_00']])
fig,axs = plt.subplots(*cols.shape)
mae = np.zeros(np.r_[cols.shape,4])
cold = np.zeros_like(mae)
heat = np.zeros_like(mae)
for i in range(cols.shape[0]):
	for j in range(cols.shape[1]):
		s = cols[i,j]
		t = Tm[s]
		B = t-T
		dz = Z['d03_op' if z.search(s) else s]-sta['elev']

		plt.sca(axs[i,j])
		axs[i,j].set_title(cols[i,j])

		for h,f in enumerate([t, t+0.0065*dz, t-B.mean(), t-pd.rolling_mean(B,7*24,freq='1H',min_periods=1)]):
			y = f-T
			mae[i,j,h] = abs(y).mean().mean()
			k = gaussian_kde(y.stack().dropna())

			p = pd.Panel({0: T.groupby(T.index.date).min(), 1:f.groupby(f.index.date).min()}).to_frame()
			a,b,c,d,n = abcd(p[p<hh.K])
			cold[i,j,h] = a/(a+b+c)

			p = pd.Panel({0: T.groupby(T.index.date).max(), 1:f.groupby(f.index.date).max()}).to_frame()
			a,b,c,d,n = abcd(p[p>hh.K+30])
			heat[i,j,h] = a/(a+b+c)
			# lab = 'MAE {:.3f}; cold {:.3f}; heat {:.3f}'.format(mae[i,j,h],cold[i,j,h],heat[i,j,h])
			# plt.plot(x,k(x),label=lab)
			plt.plot(x,k(x))
			# plt.legend()

		plt.grid()


fig,axs = plt.subplots(*cols.shape)
cy = axs[0,0]._get_lines.prop_cycler
col = [next(cy)['color'] for i in range(4)]
x = np.arange(4)
for i in range(cols.shape[0]):
	for j in range(cols.shape[1]):
		axs[i,j].set_title(cols[i,j])
		plt.sca(axs[i,j])
		p = np.r_[mae[i:i+1,j],cold[i:i+1,j],heat[i:i+1,j]]
		plt.bar(x,p[0,:],color=col)
		axs[i,j].set_ylim((0,5.1))
		ax = axs[i,j].twinx()
		ax.bar(x+5,p[1,:],color=col)
		ax.bar(x+10,p[2,:],color=col)
		ax.set_ylim((0,0.81))
		axs[i,j].set_xticks([1.5,6.5,11.5])
		if i==2: axs[i,j].set_xticklabels(['MAE','TS < 0 C', 'TS > 30C'])
		else: axs[i,j].set_xticklabels([])
		if j>0: axs[i,j].set_yticklabels([])
		if j<2: ax.set_yticklabels([])
		if i==1:
			if j==0: axs[1,0].set_ylabel('MAE')
			if j==2: ax.set_ylabel('TS')


