#!/usr/bin/env python
import requests, csv
from io import StringIO
from dateutil.parser import parse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from helpers import CEAZAMetTZ

# url = 'http://192.168.5.2/ws/pop_ws.php'		# from inside
url = 'http://www.ceazamet.cl/ws/pop_ws.php'  # from outside
raw_url = 'http://www.ceazamet.cl/ws/sensor_raw_csv.php'

trials = 10

class Base(object):
    earliest = datetime(2003,1,1)
    def __init__(self, elev, first):
        self.elev = elev
        try:
            self.first = parse(first)
        except:
            self.first = self.earliest
        self.last = datetime.utcnow() - timedelta(hours=4)

    def __repr__(self):
        return self.name


class Station(Base):
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

    def __init__(self, code, name, lon, lat, elev, first=None):
        super(Station, self).__init__(elev, first)
        self.code = code
        self.name = name
        self.lon = lon
        self.lat = lat

    @staticmethod
    def DataFrame(stations):
        df = pd.DataFrame.from_items(
            [(c, (s.name, float(s.lon), float(s.lat), float(s.elev), s.first))
             for c, s in stations.items()],
            columns = ['full', 'lon', 'lat', 'elev', 'first'],
            orient = 'index')
        df.index.name = 'code'
        return df

class Field(Base):
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

    def __init__(self, sensor, code, name, unit, elev, first):
        super(Field, self).__init__(elev, first)
        self.field = sensor
        self.sensor_code = code
        self.name = name
        self.unit = unit

    @staticmethod
    def DataFrame(stations):
        df = pd.DataFrame.from_items(
            [(f.sensor_code, (c, f.field, f.name, f.unit, f.elev))
             for c, s in stations.items() for f in s.fields],
            columns = ['station_code', 'field', 'full', 'unit', 'elev'],
            orient = 'index'
        )
        df.index.name = 'sensor_code'
        return df

def fetch(station, f, from_date=datetime(2000, 1, 1)):
    s = requests.Session()
    s.headers.update({'Host': 'www.ceazamet.cl'})
    cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
    params = {
        'fn': 'GetSerieSensor',
        'interv': 'hora',
        'valor_nan': 'nan',
        's_cod': f.name,
        'fecha_inicio': from_date.strftime('%Y-%m-%d'),
        'fecha_fin': datetime.utcnow().strftime('%Y-%m-%d')
        }
    for trial in range(trials):
        r = s.get(url, params=params)
        if not r.ok:
            continue
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
            print(r.text)
        else:
            d.columns = pd.MultiIndex.from_arrays(
                (np.repeat(station.name, 4), np.repeat(f.field, 4),
                 np.repeat(f.name, 4), np.repeat(f.elev, 4),
                 np.array(cols[1:])),
                names=['station', 'field', 'code', 'elev', 'aggr'])
            print(u'fetched {} from station {}'.format(f.field,
                                                   station.name))
            break
    try:
        return d
    except:
        return None


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


def get_stations(sta=None):
    with requests.Session() as s:
        s.headers.update({'Host': 'www.ceazamet.cl'})
        for trial in range(trials):
            req = s.get(url, params=Station.params)
            if not req.ok:
                continue
            io = StringIO(req.text)
            stations = {l[0]: Station(*l[:6]) for l in csv.reader(io) if l[0][0] != '#'}
            io.close()

        if sta is not None:
            stations = {c: st for c, st in stations.items() if c not in sta.index}

        for c, st in stations.items():
            params = Field.params.copy()
            params['e_cod'] = c
            for trial in range(trials):
                print(st.name)
                req = s.get(url, params=params)
                if not req.ok:
                    continue
                io = StringIO(req.text)
                st.fields = [
                    Field(*l[:6]) for l in csv.reader(io) if l[0][0] != '#'
                ]
                io.close()
                break
        return stations


def get_field(field, sta, field_table, from_date=datetime(2000, 1, 1), raw=False):
    D = {}
    for c, st in sta.iterrows():
        fields = field_table[field_table['station_code']==c][field_table['field']==field]
        if len(fields):
            for s, f in fields.iterrows():
                D[(c, s)] = fetch_raw(st, f, from_date) if raw else fetch(st, f, from_date)
        else:
            print("{} doesn't have {}".format(st.name, field))
    return D if raw else pd.concat(D.values(), 1).sort_index(axis=1)


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


def fetch_raw(station, f, from_date=datetime(2000, 1, 1)):
    """
    Needs to be called with a single Field instance.
    """
    s = requests.Session()
    params = {'fi': from_date.strftime('%Y-%m-%d'),
              'ff': datetime.utcnow().strftime('%Y-%m-%d'),
              's_cod': f.name}
    for trial in range(trials):
        r = s.get(raw_url, params=params)
        if not r.ok:
            continue
        try:
            reader = Reader(r.text)
        except StopIteration:
            print ('no data for field {} from station {}'.format(f.field, station.full))
            return None
        try:
            d = pd.read_csv(
                reader,
                index_col = 1,
                parse_dates = True
            )
        except:
            print('error: {} from station {}'.format(f.field, station.full))
        else:
            # hack for lines from webservice ending in comma - pandas adds additional column
            # and messes up names
            cols = [x.lstrip() for x in d.columns]
            d = d.iloc[:,1:4]
            d.columns = pd.MultiIndex.from_arrays(
                np.r_['0,2', np.repeat([[station.name], [f.field], [f.name], [f.elev]], 3, 1), cols[-3:]],
                names=['station', 'field', 'code', 'elev', 'aggr']
            )
            reader.close()
            print('fetched {} from station {}'.format(f.field, station.full))
            break
    return d


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
