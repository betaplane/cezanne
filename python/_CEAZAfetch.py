#!/usr/bin/env python
import requests, csv
from io import StringIO as strio
from datetime import datetime
import pandas as pd
import numpy as np
from CEAZAMet import Station, Field


"""
A wrapper around the CEAZAMet core classes Station, Field to obtain the data either as 
pandas DataFrame or as the raw csv. Start with creating a root object: root('inside').
"""

Trials = 10

class mixin(object):
	@property
	def url(self):
		return self._root.url
	@property
	def session(self):
		return self._root.session

class station(Station, mixin):
	def __init__(self,root,*args):
		super().__init__(*args)
		self._root = root
	
	@property
	def fields(self):
		try: 
			return self._fields
		except AttributeError:
			params = Field.params.copy()
			params.update([('e_cod',self.code)])
			tr = Trials
			while tr:
				r = self.session.get(self.url, params=params)
				with strio(r.text) as io:
					self.text = [l for l in csv.reader(io)]
				try: self._fields = [field(self,*l[:7]) for l in self.text if l[0][0]!='#']
				except: tr -= 1
				else: break
			return self._fields


class data(object):
	def __init__(self,field,io):
		self._field = field
		io.seek(0)
		p = 0
		while next(io)[0]=='#':
			n = p
			p = io.tell()
		next(io) # check for 'Este servicio ' error - should throw StopIteration exception here
		io.seek(n+1)
		self._io = io
	def __del__(self):
		try: self._io.close()
		except: pass
	
	@property
	def DataFrame(self):
		try:
			return self._dataframe
		except AttributeError:
			cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
			d = pd.read_csv(self._io, index_col=0, parse_dates=True, usecols=cols)
			d.columns = pd.MultiIndex.from_arrays((
				np.repeat(self._field._root.code,4),
				np.repeat(self._field.field,4), 
				np.repeat(self._field.sensor_code,4), 
				np.repeat(float(self._field.elev),4), 
				np.array(cols[1:])
			), names=['station','field','code','elev','aggr'])
			try: self._dataframe = self._dataframe.join(d, how="outer")
			except AttributeError: self._dataframe = d
			if hasattr(self,'_csv'): self._io.close()
			return self._dataframe
	
	@property
	def csv(self):
		try:
			return self._csv
		except AttributeError:
			self._io.seek(0)
			self._csv = [l for l in self._io]
			if hasattr(self,'_dataframe'): self._io.close()
			return self._csv


class field(Field, mixin):
	def __init__(self,root,*args):
		super().__init__(*args)
		self._root = root

	def data(self, date=None):
		try: 
			return self._data
		except AttributeError:
			params = {
				'fn':'GetSerieSensor', 'interv':'hora', 'valor_nan':'nan', 'fecha_inicio': '2003-01-01', 
				's_cod': self.sensor_code, 'fecha_fin': datetime.now().strftime('%Y-%m-%d')
			}
			if date is not None:
				params['fecha_inicio'] = date.strftime('%Y-%m-%d')
			elif isinstance(self.first,datetime):
				params['fecha_inicio'] = self.first.strftime('%Y-%m-%d')
			tr = Trials
			while tr:
				r = self.session.get(self.url, params=params)
				try: self._data = data(self, strio(r.text))
				except: tr -= 1
				else: break
			return self._data


class root(object):
	inside = 'http://192.168.5.2/ws/pop_ws.php'
	outside = 'http://www.ceazamet.cl/ws/pop_ws.php'
	def __init__(self, from_where):
		self.url = getattr(self,from_where)
		self.session = requests.Session()
		self.session.headers.update({'Host':'www.ceazamet.cl'})
	@property
	def stations(self):
		try:
			return self._stations
		except AttributeError:
			tr = Trials
			while tr:
				r = self.session.get(self.url, params=Station.params)
				with strio(r.text) as io:
					self.text = [l for l in csv.reader(io)]
				try: self._stations = [station(self,*l[:7]) for l in self.text if l[0][0]!='#']
				except: tr -= 1
				else: break
			return self._stations

	def get_field(self, field, format='DataFrame', data=None):
		for st in self.stations:
			fields = [f for f in st.fields if f.field==field]
			if data is None: date = None
			else:
				n = data.xs('data_pc',level='aggr',axis=1)
				date = n.index[n[n>0].any(1)][-1].date()
			for f in fields:
				d = getattr(f.data(date=date), format)
				try: D = D.join(d, how="outer")
				except NameError: D = d
				print('fetched {} from station {}'.format(f.field, st.name))
			if not fields:
				print("{} doesn't have {}".format(st.name, field))
		return D.sort_index(axis=1)

	

if __name__ == "__main__":
	pass