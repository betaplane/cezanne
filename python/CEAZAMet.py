#!/usr/bin/env python
import requests, csv
from io import StringIO
from dateutil.parser import parse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from helpers import CEAZAMetTZ

# url = 'http://192.168.5.2/ws/pop_ws.php'		# from inside



class FetchError(Exception):
    pass

class NoNewStationError(Exception):
    pass

class __Reader(StringIO):
    def __init__(self, str):
        super(Reader, self).__init__(str)
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

    def reset(self):
        self.seek(self.start)

class Fetcher(object):
    trials = range(10)
    url = 'http://www.ceazamet.cl/ws/pop_ws.php'
    raw_url = 'http://www.ceazamet.cl/ws/sensor_raw_csv.php'

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

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Host': 'www.ceazamet.cl'})

    def __del__(self):
        self.session.close()

    def fetch(self, code, from_date=datetime(2003, 1, 1), session=None):
        cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
        params = {
            'fn': 'GetSerieSensor',
            'interv': 'hora',
            'valor_nan': 'nan',
            's_cod': code,
            'fecha_inicio': from_date.strftime('%Y-%m-%d'),
            'fecha_fin': datetime.utcnow().strftime('%Y-%m-%d')
            }
        for trial in self.trials:
            r = self.session.get(self.url, params=params)
            if not r.ok:
                continue
            reader = __Reader(r.text)
            try:
                d = pd.read_csv(
                    reader, index_col=0, parse_dates=True, usecols=cols)
            except:
                raise FetchError(r.url)
            else:
                reader.close()
                break
        try:
            return d
        except:
            return None


    def fetch_raw(self, code, from_date=datetime(2003, 1, 1)):
        params = {'fi': from_date.strftime('%Y-%m-%d'),
                  'ff': datetime.utcnow().strftime('%Y-%m-%d'),
                  's_cod': code}
        for trial in self.trials:
            r = self.session.get(self.raw_url, params=params)
            if not r.ok:
                continue
            reader = __Reader(r.text)
            try:
                d = pd.read_csv(
                    reader,
                    index_col = 1,
                    parse_dates = True
                )
            except:
                raise FetchError(r.url)
            else:
                # hack for lines from webservice ending in comma - pandas adds additional column
                # and messes up names
                cols = [x.lstrip() for x in d.columns][-3:]
                d = d.iloc[:,1:4]
                d.columns = cols
                reader.close()
                break
        return d


    def get_stations(self, sta=None):
        with requests.Session() as s:
            for trial in self.trials:
                req = self.session.get(url, params=self.station)
                if not req.ok:
                    continue
                with StringIO(req.text) as sio:
                    try:
                        self.stations = [(l[0], l[1:6]) for l in csv.reader(sio) if l[0][0] != '#']
                    except:
                        print(trial)
                    else:
                        break
            if sta is not None:
                self.stations = [(c, st) for c, st in self.stations if c not in sta.index]
            if len(stations) == 0:
                raise NoNewStationException
            stations = pd.DataFrame.from_items(
                self.stations,
                columns = ['full', 'lon', 'lat', 'elev', 'first'],
                orient='index'
            )
            stations.index.name = 'station'

            self.fields = []
            for c, st in stations.iterrows():
                params = self.field.copy()
                params['e_cod'] = c
                for trial in self.trials:
                    print(st.name)
                    req = self.session.get(url, params=params)
                    if not req.ok:
                        continue
                    with StringIO(req.text) as sio:
                        try:
                            fields.extend(
                                [((c, l[0], l[1]), l[2:6]) for l in csv.reader(sio) if l[0][0] != '#']
                            )
                        except:
                            print(trial)
                        else:
                            break

            fields = pd.DataFrame.from_items(
                self.fields,
                columns = ['full', 'unit', 'elev', 'first'],
                orient = 'index'
            )
            fields.index = pd.MultiIndex.from_tuples(
                fields.index.tolist(),
                names = ['station', 'field', 'sensor_code']
            )
            return stations, fields.sort_index()


    def get_field(self, field, field_table, from_date=datetime(2003, 1, 1), raw=False):
        self.data = {}
        for c, f in field_table.loc[pd.IndexSlice[:, field, :], :].iterrows():
            try:
                df = self.fetch_raw(c[2], from_date) if raw else self.fetch(c[2], from_date)
            except FetchError as fe:
                print(fe)
            else:
                i = df.columns.size
                df.columns = pd.MultiIndex.from_arrays(
                    np.r_[np.repeat([c[0], field, c[2], f.elev], i), df.columns].reshape((-1, i)),
                    names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
                )
                print('fetched {} ({}) from {}'.format(field, c[2], c[0]))
                self.data[c[2]] = df
        return self.data if raw else pd.concat(self.data.values(), 1).sort_index(axis=1)


def stations_with(stations, **kw):
    S = set(stations)
    for k, v in kw.items():
        S.intersection_update(
            [s for s in stations for f in s.fields if getattr(f, k) == v])
    return list(S)


def mult(df):
    s = df.columns.get_level_values('station').tolist()
    for i in set(s):
        s.remove(i)
    return s


def get_interval(stations):
    iv = pd.Series(index=[s.code for s in stations], name='interval')
    s = requests.Session()
    s.headers.update({'Host': 'www.ceazamet.cl'})
    for st in stations:
        date = st.first
        while True:
            ds = date.strftime('%Y-%m-%d')
            params = {'s_cod': st.fields[0].sensor_code, 'fi': ds, 'ff': ds}
            r = s.get(raw_url, params=params)
            if not r.ok:
                continue
            io = csv.reader(StringIO(r.text))
            try:
                dt = np.diff(
                    np.array(
                        [l[1] for l in io if l[0][0] != '#'][:2],
                        dtype='datetime64'))[0].astype(int)
            except:
                date += timedelta(days=1)
                print('{} - {}'.format(st.name, date))
            else:
                iv[st.code] = dt
                print('{} - {}'.format(st.name, dt))
                break
    return iv



def field_table(stations):
    D = {}
    for st in stations:
        d = {}
        for F in st.fields:
            try:
                f = d[F.field]
                f[len(f)] = F.sensor_code
            except:
                d[F.field] = {0: F.sensor_code}
        D[st.code] = d
    return pd.Panel(D)


if __name__ == "__main__":
    import json, jsonpickle
    # 	from netCDF4 import Dataset
    # 	from glob import glob
    # 	stations = get_stations()
    # 	with open('data/stations.json','w') as f:
    # 		json.dump(jsonpickle.encode(stations),f)
    with open('data/stations.json') as f:
        stations = jsonpickle.decode(json.load(f))

    # Puerto Williams
    stations = [s for s in stations if s.code != 'CNPW']

    df = get_field('ts_c', stations)
    with pd.HDFStore('data/station_data.h5') as store:
        store['ts_c'] = df
