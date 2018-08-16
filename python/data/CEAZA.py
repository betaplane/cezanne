"""
CEAZAMet stations webservice
----------------------------
"""
#!/usr/bin/env python
import requests, csv, os
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from traitlets.config.configurable import Configurable
from traitlets import Unicode, Instance, Dict
from importlib import import_module


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


class CEAZAMet(Configurable):
    """Class to download data from CEAZAMet webservice. Main reason for having a class is
    to be able to reference the data (Downloader.data) in case something goes wrong at some point.
    """
    url = Unicode('').tag(config = True)
    raw_url = Unicode('').tag(config = True)
    from_date = Instance(datetime, (2003, 1, 1))
    field = Dict().tag(config = True)
    station = Dict().tag(config = True)
    data = Dict().tag(config = True)

    def __init__(self, trials=10, max_workers=16):
        loader = import_module('traitlets.config.loader')
        super().__init__(
            config = loader.PyFileConfigLoader(
                os.path.expanduser('~/Dropbox/work/config.py')).load_config()
        )
        self.trials = range(trials)
        self.max_workers = max_workers

    def _get(self, f, from_date, raw):
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


    def get_field(self, var, var_table, from_date=None, raw=False):
        """Collect data from CEAZAMet webservice, for one variable type but all stations.

        :param var: variable code to be collected (e.g. 'ta_c')
        :param var_table: pandas.DataFrame with field metadata as constructed by get_stations()
        :param from_date: initial date from which onward to request data
        :param raw: False (default) or True whether raw data should be collected
        :returns: data for one variable and all stations given by var_table
        :rtype: :class:`~pandas.DataFrame` or :obj:`dict` of DataFrames if raw==True

        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            self.data = [exe.submit(self._get, c, from_date=from_date, raw=raw) for c in
                         var_table.xs(var, 0, 'field', False).iterrows()]

        data = dict([d.result() for d in as_completed(self.data) if d.result() is not None])
        return data if raw else pd.concat(data.values(), 1).sort_index(axis=1)

    def get_fields_by_station(self, station, var_table, from_date=None, raw=False):
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            self.data = [exe.submit(self._get, c, from_date=from_date, raw=raw) for c in
                         var_table.xs(station, 0, 'station', False).iterrows()]
        data = [d.result()[1] for d in as_completed(self.data) if d.result() is not None]
        return pd.concat(data, 1).sort_index(1)

    def fetch(self, code, from_date=None):
        from_date = self.from_date if from_date is None else from_date
        cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
        params = self.data.update({
            's_cod': code,
            'fecha_inicio': from_date.strftime('%Y-%m-%d'),
            'fecha_fin': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d'),
        })
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

    def get_stations(self, sta=None, fields=True):
        """Query CEAZA webservice for a list of the stations (and all available meteorological variables for each field if ``field=True``) and return :class:`DataFrame(s)<pandas.DataFrame>` with the data.

        :param sta: existing 'stations' DataFrame to update
        :type sta: :class:`~pandas.DataFrame`
        :param fields: whether or not to return a 'fields' DataFrame (if ``True``, a tuple of (stations, fields) DataFrames is returned)
        :type fields: :obj:`bool`
        :returns: 'stations' (and optionally 'fields') DataFrame(s)
        :rtype: :class:`~pandas.DataFrame` or :obj:`tuple` of two DataFrames

        """
        for trial in self.trials:
            req = requests.get(self.url, params=self.station)
            if not req.ok:
                continue
            with StringIO(req.text) as sio:
                try:
                    self.stations = [(l[0], l[1: 7]) for l in csv.reader(sio) if l[0][0] != '#']
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
            columns = ['full', 'lon', 'lat', 'elev', 'first', 'last'],
            orient='index'
        )
        stations.index.name = 'station'

        if not fields:
            return stations.sort_index()

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
