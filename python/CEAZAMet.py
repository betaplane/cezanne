#!/usr/bin/env python
import requests, csv
from io import StringIO
from dateutil.parser import parse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from timeit import default_timer as timer
from collections import Counter

# url = 'http://192.168.5.2/ws/pop_ws.php'		# from inside



class FetchError(Exception):
    pass

class NoNewStationError(Exception):
    pass

class _Reader(StringIO):
    def __init__(self, str):
        super(_Reader, self).__init__(str)
        p = 0
        while True:
            try:
                l = next(self)
            except StopIteration:
                raise FetchError(str)
            if l[0] != '#':
                break
            self.start = p + l.find(':') + 1
            p = self.tell()
        self.seek(self.start)


class Downloader(object):
    """Class to download data from CEAZAMet webservice. Main reason for having a class is
    to be able to reference the data (Downloader.data) in case something goes wrong at some point.
    """
    trials = range(10)
    url = 'http://www.ceazamet.cl/ws/pop_ws.php'
    raw_url = 'http://www.ceazamet.cl/ws/sensor_raw_csv.php'
    max_workers = 16
    from_date = datetime(2003, 1, 1)

    field = {
        'fn': 'GetListaSensores',
        'p_cod': 'ceazamet',
        'c0': 'tm_cod',
        'c1': 's_cod',
        'c2': 'tf_nombre',
        'c3': 'um_notacion',
        'c4': 's_altura',
        'c5': 's_primera_lectura',
        'c6': 's_ultima_lectura'
    }

    station = {
        'fn': 'GetListaEstaciones',
        'p_cod': 'ceazamet',
        'c0': 'e_cod',
        'c1': 'e_nombre',
        'c2': 'e_lon',
        'c3': 'e_lat',
        'c4': 'e_altitud',
        'c5': 'e_primera_lectura',
        'c6': 'e_ultima_lectura'
    }

    def get_field(self, var, var_table, from_date=None, raw=False):
        """Collect data from CEAZAMet webservice, for one variable type but all stations.

        :param var: variable code to be collected (e.g. 'ta_c')
        :param var_table: pandas.DataFrame with field metadata as constructed by get_stations()
        :param from_date: initial date from which onward to request data
        :param raw: False (default) or True whether raw data should be collected
        :returns: data for one variable and all stations given by var_table
        :rtype: pandas.DataFrame or dict with DataFrames if raw==True

        """
        self.data = {}
        def get(f):
            if from_date is None:
                if 'last' in f[1]:
                    day = f[1]['last'].to_pydatetime()
                    day = day - timedelta(days = 1) if day == day else self.from_date
                else:
                    day = self.from_date
            else:
                day = from_date
            try:
                df = self.fetch_raw(f[0][2], day) if raw else self.fetch(f[0][2], day)
            except FetchError as fe:
                print(fe)
            else:
                i = df.columns.size
                df.columns = pd.MultiIndex.from_arrays(
                    np.r_[
                        np.repeat([f[0][0], f[0][1], f[0][2], f[1]['elev']], i),
                        df.columns
                    ].reshape((-1, i)),
                    names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
                )
                print('fetched {} from {}'.format(f[0][2], f[0][0]))
                return f[0][2], df.sort_index(1)

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            self.data = [exe.submit(get, c) for c in
                         var_table.xs(var, 0, 'field', False).iterrows()]

        data = dict([d.result() for d in as_completed(self.data) if d.result() is not None])
        return data if raw else pd.concat(data.values(), 1).sort_index(axis=1)

    def fetch(self, code, from_date=None):
        from_date = self.from_date if from_date is None else from_date
        cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
        params = {
            'fn': 'GetSerieSensor',
            'interv': 'hora',
            'valor_nan': 'nan',
            's_cod': code,
            'fecha_inicio': from_date.strftime('%Y-%m-%d'),
            'fecha_fin': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d')
            }
        for trial in self.trials:
            r = requests.get(self.url, params=params)
            if not r.ok:
                continue
            reader = _Reader(r.text)
            try:
                d = pd.read_csv(
                    reader, index_col=0, parse_dates=True, usecols=cols)
                reader.close()
            except:
                raise FetchError(r.url)
            else:
                return d.astype(float) # important

    def fetch_raw(self, code, from_date=None):
        from_date = self.from_date if from_date is None else from_date
        params = {'fi': from_date.strftime('%Y-%m-%d'),
                  'ff': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d'), 
                  's_cod': code}
        for trial in self.trials:
            r = requests.get(self.raw_url, params=params)
            if not r.ok:
                continue
            reader = _Reader(r.text)
            try:
                d = pd.read_csv(
                    reader,
                    index_col = 1,
                    parse_dates = True
                )
                reader.close()
            except:
                raise FetchError(r.url)
            else:
                # hack for lines from webservice ending in comma - pandas adds additional column
                # and messes up names
                cols = [x.lstrip() for x in d.columns][-3:]
                d = d.iloc[:,1:4]
                d.columns = cols
                return d.astype(float) # important

    def get_stations(self, sta=None):
        for trial in self.trials:
            req = requests.get(self.url, params=self.station)
            if not req.ok:
                continue
            with StringIO(req.text) as sio:
                try:
                    self.stations = [(l[0], l[1:6]) for l in csv.reader(sio) if l[0][0] != '#']
                except:
                    print('attempt #{}'.format(trial))
                else:
                    break

        if sta is not None:
            self.stations = [(c, st) for c, st in self.stations if c not in sta.index]

        if len(self.stations) == 0:
            raise NoNewStationError

        stations = pd.DataFrame.from_items(
            self.stations,
            columns = ['full', 'lon', 'lat', 'elev', 'first'],
            orient='index'
        )
        stations.index.name = 'station'

        def get(st):
            params = self.field.copy()
            params['e_cod'] = st[0]
            for trial in self.trials:
                print(st[1].full)
                req = requests.get(self.url, params=params)
                if not req.ok:
                    continue
                with StringIO(req.text) as sio:
                    try:
                        return [((st[0], l[0], l[1]), l[2:6])
                                  for l in csv.reader(sio) if l[0][0] != '#']
                    except:
                        print('attempt #{}'.format(trial))

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            fields = [exe.submit(get, s) for s in stations.iterrows()]

        self.fields = [f for g in as_completed(fields) for f in g.result()]

        fields = pd.DataFrame.from_items(
            self.fields,
            columns = ['full', 'unit', 'elev', 'first'],
            orient = 'index'
        )
        fields.index = pd.MultiIndex.from_tuples(
            fields.index.tolist(),
            names = ['station', 'field', 'sensor_code']
        )
        return stations.sort_index(), fields.sort_index()
