#!/usr/bin/env python
import numpy as np
import pandas as pd
import pygrib
from datetime import datetime


p = [1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50]

def ungrib(path):
	def time(m):
		s = str(m['validityDate'])
		t = str(m['validityTime'])
		return datetime(int(s[:4]),int(s[4:6]),int(s[6:8]),int(t[:2]))

	t = {}
	for m in pygrib.open(path):
		try: 
			t[time(m)][m['level']] = m.data()[0]
		except KeyError:
			t[time(m)] = {m['level']:m.data()[0]}
	lat,lon = m.latlons()
	
	time = []
	data = []
	for d in sorted(t.items(), key=lambda x:x[0]):
		time.append(d[0])
		u = []
		level = []
		for l in sorted(d[1].items(), key=lambda x:x[0], reverse=True):
			u.append(l[1])
			level.append(l[0])
		data.append(u)
	
	return np.array(data), np.array(time,dtype='datetime64[h]'), np.array(level), lat, lon

def align(x1,x2,t1,t2):
	s = set(t1).intersection(t2)
	return x1[[np.where(t1==x)[0][0] for x in s],:,:,:], x2[[np.where(t2==x)[0][0] for x in s],:,:,:]