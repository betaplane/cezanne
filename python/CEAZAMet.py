#!/usr/bin/env python
import requests, csv, re
from io import StringIO
from dateutil.parser import parse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from helpers import CEAZAMetTZ

# url = 'http://192.168.5.2/ws/pop_ws.php'		# from inside
url = 'http://www.ceazamet.cl/ws/pop_ws.php'  # from outside
raw_url = 'http://www.ceazamet.cl/ws/sensor_raw_csv.php'


class Station(object):
    params = {
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

    def __init__(self, code, name, lon, lat, elev, first, last):
        self.code = code
        self.name = name
        self.lon = lon
        self.lat = lat
        self.elev = elev
        self.first = parse(first)
        self.last = parse(last)

    def __repr__(self):
        return self.name


class Field(object):
    earliest = datetime(2003,1,1)
    params = {
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

    def __init__(self, sensor, code, name, unit, elev, first, last):
        self.field = sensor
        self.sensor_code = code
        self.name = name
        self.unit = unit
        self.elev = elev
        try:
            self.first = parse(first)
        except:
            self.first = self.earliest
        try:
            self.last = parse(last)
        except:
            self.last = datetime.utcnow() - timedelta(hours=4)

    def __repr__(self):
        return self.name



def fetch(station, field=None):
    s = requests.Session()
    s.headers.update({'Host': 'www.ceazamet.cl'})
    cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
    params = {'fn': 'GetSerieSensor', 'interv': 'hora', 'valor_nan': 'nan'}
    for f in ([field] if field else station.fields):
        params.update({
            's_cod': f.sensor_code,
            'fecha_inicio': f.first.strftime('%Y-%m-%d'),
            'fecha_fin': f.last.strftime('%Y-%m-%d')
        })
        while True:
            r = s.get(url, params=params)
            io = StringIO(r.text)
            p = 0
            while next(io)[0] == '#':
                n = p
                p = io.tell()
            io.seek(n + 1)
            try:
                d = pd.read_csv(
                    io, index_col=0, parse_dates=True, usecols=cols)
            except:
                pass
            else:
                d.columns = pd.MultiIndex.from_arrays(
                    (np.repeat(station.code, 4), np.repeat(f.field, 4),
                     np.repeat(f.sensor_code, 4), np.repeat(f.elev, 4),
                     np.array(cols[1:])),
                    names=['station', 'field', 'code', 'elev', 'aggr'])
                try:
                    D = D.join(d, how="outer")
                except NameError:
                    D = d
                io.close()
                print(u'fetched {} from station {}'.format(f.field,
                                                           station.name))
                break
    return D

class Reader(StringIO):
    def __init__(self, str):
        super(Reader, self).__init__(str)
        p = 0
        while True:
            l = next(self)
            if l[0] != '#':
                break
            self.start = p + l.find(':') + 1
            p = self.tell()
        self.seek(self.start)

    def reset(self):
        self.seek(self.start)


def get_stations():
    s = requests.Session()
    s.headers.update({'Host': 'www.ceazamet.cl'})
    req = s.get(url, params=Station.params)
    io = StringIO(req.text)
    stations = [Station(*l[:7]) for l in csv.reader(io) if l[0][0] != '#']
    io.close()

    def sensor(station):
        params = Field.params.copy()
        params.update([('e_cod', station.code)])
        return params

    for st in stations:
        while True:
            print(st.name)
            req = s.get(url, params=sensor(st))
            io = StringIO(req.text)
            try:
                st.fields = [
                    Field(*l[:7]) for l in csv.reader(io) if l[0][0] != '#'
                ]
            except:
                pass
            else:
                break
            io.close()
    return stations


def get_field(field, stations, raw=False):
    for st in stations:
        fields = [f for f in st.fields if f.field == field]
        for f in fields:
            d = fetch_raw(st,f) if raw else fetch(st, f)
            try:
                D = D.join(d, how="outer")
            except NameError:
                D = d
        if not fields:
            print(u"{} doesn't have {}".format(st.name, field))
    return D.sort_index(axis=1)


def stations_with(stations, **kw):
    S = set(stations)
    for k, v in kw.items():
        S.intersection_update(
            [s for s in stations for f in s.fields if getattr(f, k) == v])
    return list(S)


def all_fields(stations):
    return pd.DataFrame(
        [(f.field, f.name, f.unit, f.elev, s.code, s.name)
         for s in stations for f in s.fields],
        columns=[
            'field', 'full_name', 'unit', 'elev', 'station_code', 'station'
        ])


def sta2df(stations):
    return pd.DataFrame.from_items(
        [(s.code, (s.name, float(s.lon), float(s.lat), float(s.elev)))
         for s in stations],
        orient='index',
        columns=['name', 'lon', 'lat', 'elev'])


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


def fetch_raw(station, field=None):
    s = requests.Session()
    params = {'fi':'2000-01-01', 'ff':'2020-01-01'}
    for f in ([field] if field else station.fields):
        params.update({'s_cod': f.sensor_code})
        trials = 10
        while trials:
            trials -= 1
            r = s.get(raw_url, params=params)
            reader = Reader(r.text)
            try:
                d = pd.read_csv(
                    reader,
                    index_col = 1,
                    parse_dates = True
                )
            except:
                print('error: {} from station {}'.format(f.field, station.name))
            else:
                # hack for lines from webservice ending in comma - pandas adds additional column
                # and messes up names
                cols = d.columns
                d = d.iloc[:,1:4]
                d.columns = pd.MultiIndex.from_arrays(
                    np.r_['0,2', np.repeat([[station.code], [f.field], [f.sensor_code], [f.elev]], 3, 1), cols[-3:]],
                    names=['station', 'field', 'code', 'elev', 'aggr']
                )
                try:
                    D = D.join(d, how="outer")
                except NameError:
                    D = d
                reader.close()
                print('fetched {} from station {}'.format(f.field, station.name))
                break
    return D



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
