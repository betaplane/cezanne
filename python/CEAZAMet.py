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

    def get_field(self, field, field_table, from_date=None, raw=False):
        from_date = self.from_date if from_date is None else from_date
        self.data = {}
        def get(f):
            try:
                df = self.fetch_raw(f[0][2], from_date) if raw else self.fetch(f[0][2], from_date)
            except FetchError as fe:
                print(fe)
            else:
                i = df.columns.size
                df.columns = pd.MultiIndex.from_arrays(
                    np.r_[
                        np.repeat([f[0][0], f[0][1], f[0][2], f[1].elev], i),
                        df.columns
                    ].reshape((-1, i)),
                    names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
                )
                print('fetched {} from {}'.format(f[0][2], f[0][0]))
                return f[0][2], df

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            self.data = [exe.submit(get, c) for c in
                         field_table.loc[pd.IndexSlice[:, field, :], :].iterrows()]

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
            'fecha_fin': datetime.utcnow().strftime('%Y-%m-%d')
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
                return d.astype(float)

    def fetch_raw(self, code, from_date=None):
        from_date = self.from_date if from_date is None else from_date
        params = {'fi': from_date.strftime('%Y-%m-%d'),
                  'ff': datetime.utcnow().strftime('%Y-%m-%d'),
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
                return d.astype(float)

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


def time_irreg(df):
    """Check time index of DataFrame for irregularities.

    :param df: pandas.DataFrame with time index
    :returns: DataFrame with columns of irregular time intervals before and/or after a given index;
    list of irregular indexes ocurring in groups; list of 'lonely' irrgular indexes.
    :rtype: pandas.DataFrame, list of numpy.arrays, numpy.array

    """
    t = np.array(df.index, dtype='datetime64[m]')
    dt = np.diff(t).astype(float)
    # counts distinct elements in dt
    c = Counter(dt).most_common(1)[0][0]

    # look for indexes which are != the most common timestep on both sides
    d = np.r_[np.nan, dt, dt, np.nan].reshape((2,-1))
    i = (d != c).all(0) # not ok
    # DataFrame with intervals before and after timestamp
    f = pd.DataFrame(d[:,i].T, index=df.index[i])

    # look for groups of not ok indexes
    p = np.where(i)[0]
    q = np.where(np.diff(p) != 1)[0]
    q = np.r_['0', [-1], q, q , [-2]] + 1
    # groups
    g = [p[x[0]:x[1]] for x in q.reshape((2, -1)).T if np.diff(x) > 1]
    # 'lonely' indexes
    l = np.array(sorted(set(p) - set([x for y in g for x in y])))
    return f, g, l


def binning(df, start_minute, freq, label='end'):
    """Average raw data from CEAZAMet stations (fetched by CEAZAMet.Downloader.fetch_raw) into true time intervals.
    Assumptions:
    1) The (single) predominant time interval between records is taken to be the true interval over which the logger
    records/averages data.
    2) The logger's timestamp refers to the end of a 'recording interval'.
    3) If a binning boundary splits a record interval, the record's average is distributed proportionally between
    adjacent binning intervals. The max / min values are binned with that binning interval that covers the larger
    part of the record interval, since it is impossible to tell when the max / min was recorded.

    :param df: data to be averaged / binned
    :type df: pandas.DataFrame with 'avg', 'min', 'max' labels at level 'aggr' in the columns MultiIndex
    :param start_minute: minute within an hour at which a binning interval starts
    :param freq: length of binning interval (in minutes)
    :param label: at what timepoint to label the result: 'start', 'middle' or 'end' (default) of interval
    :returns: averaged and max / min values of data
    :rtype: pandas.DataFrame with same columns as input

    """
    t = np.array(df.index, dtype='datetime64[m]')
    dt = np.diff(t).astype(float)
    # counts distinct elements in dt
    c = Counter(dt).most_common(1)[0][0]

    print('record interval detected as {:d} minutes'.format(int(c)))

    # compute binning intervals
    ti = t.astype(int) - c                            # start point of record intervals
    ts = start_minute + (ti - start_minute) // freq * freq  # same label for intervals of length 'freq' from global origin
    v = ti + c - ts - freq
    k = (v > 0)               # record intervals split by end of binning intervals
    w = v[k].reshape((-1, 1)) # minutes of record intervals falling into next binning interval
    ts = ts.reshape((-1, 1))

    # for mean, split record intervals proportionally into adjacent binning intervals
    ave = df.xs('avg', 1, 'aggr', False)
    cols = ave.columns
    a = ave.as_matrix()
    aft = np.r_['1', a[k] * w / c, ts[k]]
    a[k] = a[k] * (c - w) / c
    a = np.r_['1', a, ts]
    a = np.r_['0', a, aft]
    ave = pd.DataFrame(a).groupby(a.shape[1] - 1).mean()
    ave.columns = cols

    def col(x, c):
        x.columns = pd.MultiIndex.from_tuples(x.columns, names=df.columns.names)
        return x.xs(c, 1, 'aggr', False)

    # for min and max, use the binning interval with the largest overlap with the record interval
    ts[v > c/2] = ts[v > c/2] + freq
    b = df.drop('avg', 1, 'aggr').join(pd.DataFrame(ts, index=df.index, columns=['ts'])).groupby('ts')
    D = pd.concat((col(b.min(), 'min'), ave, col(b.max(), 'max')), 1)

    lab = pd.Timedelta({'end': freq, 'middle': freq/2, 'start': 0}[label], 'm')
    D.index = pd.DatetimeIndex(np.array(D.index, dtype='datetime64[m]').astype(datetime)) + lab
    return D.sort_index(1)

def combine_raw(*args):
    def key(f, k):
        d = f[k]
        return d.columns.get_level_values('station').unique()[0], d

    X = [dict([key(x, k) for k in x.keys()]) for x in args]

    d = {}
    for k in set.union(*[set(x.keys()) for x in X]):
        d[k] = pd.concat([x.get(k, None) for x in X], 1)

    return d
