#!/usr/bin/env python
import numpy as np
import pandas as pd
import helpers as hh
import matplotlib.pyplot as plt

# relationship temp-humidity -- not useful


D = pd.HDFStore('../../data/tables/station_data.h5')
S = pd.HDFStore('../../data/tables/LinearLinear.h5')

def scapl(x,y,gr):
	ym = y.mean().as_matrix()
	d = pd.DataFrame(np.repeat([ym],b.shape[0],0) ,index=y.index, columns=y.columns)
	p = pd.Panel({'x':x,'y':y,'m':d})
	if gr==0:
		p = p.groupby(p.major_axis.date).mean().to_frame()
	else:
		p = p.groupby((p.major_axis.year,p.major_axis.month)).mean().to_frame()
	plt.scatter(p['x'],p['y'],c=p['m'])
	plt.colorbar()
	plt.grid()


T = hh.extract(D['ta_c'],'prom',1)
Tm = S['T2']
rh = hh.extract(D['rh'],'prom')
rm = S['RH2']

plt.set_cmap('gnuplot')
fig = plt.figure()
for i,d in enumerate(['d02','d03_0_00','d03_0_12']):
	b = Tm[d] - T
	plt.subplot(3,4,4*i+1)
	scapl(T,b,0)
	if i==0: plt.gca().set_title('T err vs obs, daily')
	plt.subplot(3,4,4*i+2)
	scapl(T,b,1)
	if i==0: plt.gca().set_title('T err vs obs, monthly')
	b = rm[d] - rh
	plt.subplot(3,4,4*i+3)
	scapl(rh,b,0)
	if i==0: plt.gca().set_title('RH err vs obs, daily')
	plt.subplot(3,4,4*i+4)
	scapl(rh,b,1)
	if i==0: plt.gca().set_title('RH err vs obs, monthly')
fig.show()
