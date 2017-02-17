#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import stipolate as st
import helpers as hh


D = pd.HDFStore('../data/station_data.h5')
S = pd.HDFStore('../data/LinearLinear.h5')

T = hh.extract(D['ta_c'],'prom',1)
Tm = S['T2']

def c(df,level): return df.columns.get_level_values(level)
w = D['vv_ms'].xs('prom',1,level='aggr').drop(30,1,'elev')
w.columns = w.columns.droplevel(['field','code'])
w = w.loc[:,list(set([(s,c(w[s],'elev').max()) for s in c(w,'station')]))]
w.columns = w.columns.get_level_values('station')

u = S['vv']

r2 = np.zeros((3,5))
c = np.zeros((3,5))
fig, axs = plt.subplots(3,4)
def scatter(x,y,i,j):
	global r2,c,ax
	m = pd.concat((x,y),axis=1).dropna(0,'any')
	m = m.unstack()
# 	m = m.groupby(m.index.date).mean().stack().as_matrix()
	m = m.groupby((m.index.year,m.index.month)).mean().stack().as_matrix()
# 	m = m.mean().unstack().as_matrix().T
	lm = sm.OLS(m[:,0],sm.add_constant(m[:,1]))
	r = lm.fit()
	r2[j,i] = r.rsquared_adj
	c[j,i] = r.params[1]
	axs[j,i].plot(m[:,0],m[:,1],'+')

for i,s in enumerate(['d02','d03_0_00','d03_0_12','d03_orl']):
	dt = (Tm[s]-T).stack()
	scatter(w.stack(),dt,i,0)
	axs[0,i].set_title(s)
	dw = u[s]-w
	scatter(dw.stack(),dt,i,1)
	scatter(u[s].stack(),dt,i,2)