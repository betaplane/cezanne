#!/usr/bin/env python
from netCDF4 import Dataset
import csv, glob, os
from datetime import datetime, timedelta


path = '/Volumes/datos/sata1_boletin/WRF_interanual'

start = []
sim_start = []
t_base = []
t_start = []
t_end = []
files = sorted(glob.glob(path+'/*d02*'))

for f in files:
	nc = Dataset(f)
	start.append(datetime.strptime(nc.START_DATE, '%Y-%m-%d_%H:%M:%S'))
	sim_start.append(datetime.strptime(nc.SIMULATION_START_DATE, '%Y-%m-%d_%H:%M:%S'))
	t = nc.variables['XTIME']
	t_base.append(datetime.strptime(t.units[14:], '%Y-%m-%d %H:%M:%S'))
	t_start.append(timedelta(minutes=int(t[0])) + t_base[-1])
	t_end.append(timedelta(minutes=int(t[-1])) + t_base[-1])
	


with open('file_times.csv','w') as f:
	w = csv.writer(f)
	for i,n in enumerate(files):
		w.writerow([os.path.split(n)[1],start[i],sim_start[i],t_base[i],t_start[i],t_end[i]])
		
