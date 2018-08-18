#!/usr/bin/env python
"""
CEAZAMet stations webservice
----------------------------

This module can be imported and used as a standalone command-line app. The use as module is documented in the class :class:`CEAZAMet`.

Interactive Use
===============

There are two subcommands, ``meta`` and ``data``, corresponding to the methods :meth:`get_meta` and :meth:`get_data`. They can be supplied with command-line arguments in the `IPython  <https://ipython.readthedocs.io/en/stable/config/index.html>`_ / `Jupyter <https://jupyter.readthedocs.io/en/latest/projects/config.html>`_ config style. This means that help is also available in the same style, e.g.::

    ./CEAZA.py data --help

or::

    ./CEAZA.py data --help-all

To fetch the CEAZAMet station metadata and save it in the file specified in the :attr:`CEAZAMet.station_meta` configurable, do::

    ./CEAZA.py meta

To save it in a different file, pass the file name as an command-line argument::

    ./CEAZA.py meta --Meta.file_name=filename

or with alias::

    ./CEAZA.py meta --f=filename

To fetch a particular field (e.g. 'ta_c') from all stations, do::

    ./CEAZA.py data --Field.var_name=ta_c

or with the alias::

    ./CEAZA.py data --v=ta_c

To change the file in which the results are save, pass the file name to :attr:`CEAZAMet.station_data`::

    ./CEAZA.py data --v=ta_c --f=filename

"""
import requests, csv, os
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from traitlets.config import Application
from traitlets import Unicode, Instance, Dict, Integer, Bool


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

class Field(Application):
    raw_url = Unicode('').tag(config = True)
    user = Unicode('').tag(config = True)
    from_date = Instance(datetime, (2003, 1, 1))
    var_name = Unicode('', help='Field to fetch, e.g. ta_c.').tag(config = True)
    raw = Bool(False, help='Whether to fetch the raw (as opposed to database-aggregated) data.')
    file_name = Unicode('').tag(config = True)

    aliases = Dict({'v': 'Field.var_name', 'f': 'Field.file_name'})

    def start(self):
        print(self.file_name)
        return None
        if self.raw and self.cli_config != {}:
            raise Exception('Raw saving not supported yet in command-line mode.')
        var_table = pd.read_hdf(self.file_name, 'fields').xs(self.var_name, 0, 'field', False)
        with ThreadPoolExecutor(max_workers=self.parent.max_workers) as exe:
            self.parent.data = [exe.submit(self._get, c) for c in var_table.iterrows()]

        data = dict([d.result() for d in as_completed(self.parent.data) if d.result() is not None])
        self.parent.data = data

        if not self.raw:
            data = pd.concat(data.values(), 1).sort_index(axis=1)
        if self.cli_config != {}:
            with pd.HDFStore(self.file_name, 'a') as S:
                S[self.var_name] = data
            print('Field {} fetched and saved in file {}'.format(self.var_name, self.file_name))
        else:
            return data

    def get_fields_by_station(self, station, var_table, from_date=None, raw=False):
        raise Exception("Thought this was the same as 'get_field'. Look in the repo if needed.")

    def _get(self, f):
        if 'last' in f[1]:
            day = f[1]['last'].to_pydatetime()
            day = day - timedelta(days = 1) if day == day else self.from_date
        else:
            day = self.from_date

        try:
            df = self.fetch_raw(f[0][2], day) if self.raw else self.fetch_aggr(f[0][2], day)
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

    def fetch_aggr(self, code, from_date=None):
        from_date = self.from_date if from_date is None else from_date
        cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
        params = {
            'fn': 'GetSerieSensor',
            'interv': 'hora',
            'valor_nan': 'nan',
            'user': self.user,
            's_cod': code,
            'fecha_inicio': from_date.strftime('%Y-%m-%d'),
            'fecha_fin': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d'),
        }
        for trial in range(self.parent.trials):
            r = requests.get(self.parent.url, params=params)
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
        for trial in range(self.parent.trials):
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


class Meta(Application):
    field = Dict().tag(config = True)
    field_index = Instance(slice, (0, 2))
    field_data = Instance(slice, (2, 6))
    station = Dict().tag(config = True)
    file_name = Unicode('').tag(config = True)
    get_fields = Bool(True).tag(config = True)

    aliases = Dict({'f': 'Meta.file_name'})
    flags = Dict({'n': ({'Meta': {'get_fields': False}}, "do not fetch field metadata")})

    def start(self, stations=None, interactive=False):
        params = {k: v[0] for k, v in self.station.items()}
        for trial in range(self.parent.trials):
            req = requests.get(self.parent.url, params = params)
            if not req.ok:
                continue
            with StringIO(req.text) as sio:
                try:
                    self.parent.data = [(l[0], l[1: 7]) for l in csv.reader(sio) if l[0][0] != '#']
                except:
                    print('attempt #{}'.format(trial))
                else:
                    break

        # for update, although I'm not using this currently
        if stations is not None:
            self.parent.data = [(c, st) for c, st in self.parent.data if c not in stations.index]

        if len(self.parent.data) == 0:
            raise NoNewStationError

        cols = [v[1] for k, v in sorted(self.station.items(), key=lambda k: k[0]) if k[0]=='c']
        meta = pd.DataFrame.from_items(
            self.parent.data,
            columns = cols[1:], # 0 is index
            orient='index'
        )
        meta.index.name = cols[0]
        meta.sort_index(inplace=True)

        if self.get_fields:
            params = {k: v[0] for k, v in self.field.items()}
            def get(st):
                params['e_cod'] = st[0]
                for trial in range(self.parent.trials):
                    print(st[1].full)
                    req = requests.get(self.parent.url, params = params)
                    if not req.ok:
                        continue
                    with StringIO(req.text) as sio:
                        try:
                            return [(tuple(np.r_[st[:1], l[self.field_index]]), l[self.field_data])
                                      for l in csv.reader(sio) if l[0][0] != '#']
                        except:
                            print('attempt #{}'.format(trial))

            with ThreadPoolExecutor(max_workers=self.parent.max_workers) as exe:
                field_meta = [exe.submit(get, s) for s in meta.iterrows()]

            self.parent.data = [f for g in as_completed(field_meta) for f in g.result()]

            cols = [v[1] for k, v in sorted(self.field.items(), key=lambda k: k[0]) if k[0]=='c']
            field_meta = pd.DataFrame.from_items(
                self.parent.data,
                columns = cols[self.field_data],
                orient = 'index'
            )
            field_meta.index = pd.MultiIndex.from_tuples(
                field_meta.index.tolist(),
                names = np.r_[['station'], cols[self.field_index]]
            )
            field_meta.sort_index(inplace=True)

            if self.cli_config == {}:
                return meta, field_meta
        else:
            if self.cli_config == {}:
                return meta

        with pd.HDFStore(self.file_name, mode='w') as S:
            S['stations'] = meta
            if self.get_fields:
                S['fields'] = field_meta
        print("CEAZAMet station metadata saved in file {}.".format(self.file_name))

class CEAZAMet(Application):
    """Class to download data from CEAZAMet webservice. Main reason for having a class is
    to be able to reference the data (CEAZAMet.data) in case something goes wrong at some point.

    """
    url = Unicode('').tag(config = True)

    subcommands = Dict({'meta': (Meta, 'get stations and field metadata'),
                        'data': (Field, 'get one field from all stations')})

    trials = Integer(10)
    max_workers = Integer(16)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

    def get_meta(self, stations=None, fields=True):
        """Query CEAZA webservice for a list of the stations (and all available meteorological variables for each field if ``field=True``) and return :class:`DataFrame(s)<pandas.DataFrame>` with the data.

        :param stations: existing 'stations' DataFrame to update
        :type stations: :class:`~pandas.DataFrame`
        :param fields: whether or not to return a 'fields' DataFrame (if ``True``, a tuple of (stations, fields) DataFrames is returned)
        :type fields: :obj:`bool`
        :returns: 'stations' (and optionally 'fields') DataFrame(s)
        :rtype: :class:`~pandas.DataFrame` or :obj:`tuple` of two DataFrames

        """
        app = Meta(parent=self)
        app.get_fields = fields
        return app.start(stations)

    def get_data(self):
        """Collect data from CEAZAMet webservice, for one variable type but all stations.

        :param var: variable code to be collected (e.g. 'ta_c')
        :param var_table: pandas.DataFrame with field metadata as constructed by get_stations()
        :param from_date: initial date from which onward to request data
        :param raw: False (default) or True whether raw data should be collected
        :returns: data for one variable and all stations given by var_table
        :rtype: :class:`~pandas.DataFrame` or :obj:`dict` of DataFrames if raw==True

        """
        app = Field(parent=self)
        return app.start()

if __name__ == '__main__':
    app = CEAZAMet()
    app.initialize()
    app.subapp.start()
