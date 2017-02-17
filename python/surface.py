#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import stipolate as st
import helpers as hh
import mapping as mp


D = pd.HDFStore('../data/station_data.h5')
S = pd.HDFStore('../data/LinearLinear.h5')

T = hh.extract(D['ta_c'],'prom',1)
Tm = S['T2']
b = Tm['d02']-T

nc = Dataset('../data/wrf/fx.nc')
ma = mp.basemap(nc)
nc.close()
nc = Dataset('../data/wrf/d02_2014-09-10.nc')



p = {
	'hfx': st.interp_nc(nc,'HFX',sta,map=ma)
	'qfx': st.interp_nc(nc,'QFX',sta,map=ma)
	'gfx': st.interp_nc(nc,'GRDFLX',sta,map=ma)
	'res': st.interp_nc(nc,'NOAHRES',sta,map=ma)
}

# r2 = np.zeros((3,5))
# c = np.zeros((3,2))
fig, axs = plt.subplots(2,4)
def scatter(x,y,i,j):
	global r2,c,ax
	m = pd.concat((x,y),axis=1).dropna(0,'any')
	m = m.unstack()
	if i==0:
		m = m.groupby(m.index.date).mean().stack().as_matrix()
	else:
		m = m.groupby((m.index.year,m.index.month)).mean().stack().as_matrix()
# 	m = m.mean().unstack().as_matrix().T
# 	lm = sm.OLS(m[:,0],sm.add_constant(m[:,1]))
# 	r = lm.fit()
# 	r2[j,i] = r.rsquared_adj
# 	c[j,i] = r.params[1]
	axs[i,j].plot(m[:,0],m[:,1],'+')

for i,s in enumerate(['day','month']):
	for j,k in enumerate(['hfx','gfx','qfx','res']):
		axs[i,j].set_title(k)
		scatter(p[k].stack(),b,i,j)



	