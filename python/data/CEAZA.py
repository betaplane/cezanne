#!/usr/bin/env python
"""
CEAZAMet stations webservice
----------------------------

This module can be imported and used as a standalone command-line app. The use as module is documented in the class :class:`CEAZAMet`.

Command-line use
================

There are two subcommands, ``meta`` and ``data``, corresponding to the methods :meth:`get_meta` and :meth:`get_data`. They can be supplied with command-line arguments in the `IPython  <https://ipython.readthedocs.io/en/stable/config/index.html>`_ / `Jupyter <https://jupyter.readthedocs.io/en/latest/projects/config.html>`_ config style. This means that help is also available in the same style, e.g.::

    ./CEAZA.py data --help

or::

    ./CEAZA.py data --help-all

To fetch the CEAZAMet station metadata and save it in the file specified in the :attr:`CEAZAMet.station_meta` configurable, do::

    ./CEAZA.py meta

To save it in a different file, pass the file name as an command-line argument::

    ./CEAZA.py meta --Meta.file_name=filename

or with alias::

    ./CEAZA.py meta -f filename

To fetch a particular field (e.g. 'ta_c') from all stations, do::

    ./CEAZA.py data --Field.var_name=ta_c

or with the alias::

    ./CEAZA.py data -v ta_c

To change the file in which the results are save, pass the file name to :attr:`CEAZAMet.station_data`::

    ./CEAZA.py data -v ta_c -f filename


Logging
=======

The built-in log handler for :mod:`traitlets.config` is :class:`logging.StreamHandler`. So to see logging info on the command line, do::

    ./CEAZA.py [...] --log_level=INFO 2>&1

or to redirect to a file::

    ./CEAZA.py [...] --log_level=INFO 2>logfile

.. NOTE::

    One hyphen can only be used with one-letter flags. The following are equivalent::

        ./CEAZA.py meta --Meta.filename=filename
        ./CEAZA.py meta --f=filename
        ./CEAZA.py meta -f filename

    Note that in the one-hyphen case, the **equal** sign is not needed.

.. TODO::

    * save raw data (filename = field_code, tablename = station_code)
    * rework printed info (log?)

"""
import requests, csv, os, sys
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from traitlets.config import Application
from traitlets import Unicode, Instance, Dict, Integer, Bool
from importlib import import_module
from tqdm import tqdm


class FetchError(Exception):
    pass

class NoNewStationError(Exception):
    pass

class TrialsExhaustedError(Exception):
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
    raw_url = Unicode().tag(config = True)
    user = Unicode().tag(config = True)
    from_date = Instance(datetime, (2003, 1, 1))
    var_code = Unicode('', help='Field to fetch, e.g. ta_c.').tag(config = True)
    raw = Bool(False, help='Whether to fetch the raw (as opposed to database-aggregated) data.')

    file_name = Unicode('', help='Data file name.').tag(config = True)
    """DataFrame with station locations (as returned by a call to :meth:`.get_stations`)."""

    aliases = Dict({'v': 'Field.var_code',
                    'f': 'Field.file_name',
                    'm': 'Meta.file_name',
                    'log_level': 'Application.log_level'})
    flags = Dict({'r': ({'Field': {'raw': True}}, "set raw=True")})

    def start(self, fields_table=None, from_date=None):
        if self.raw and self.cli_config != {}:
            raise Exception('Raw saving not supported yet in command-line mode.')
        if fields_table is None:
            print('Using field table from file {}'.format(self.config.Meta.file_name))
            fields_table = pd.read_hdf(self.config.Meta.file_name, 'fields').xs(self.var_code, 0, 'field', False)
        # NOTE: I'm not sure if tqdm is thread-safe
        with tqdm(total = fields_table.shape[0]) as prog:
            with ThreadPoolExecutor(max_workers=self.parent.max_workers) as exe:
                self.parent.data = [exe.submit(self._get, c, prog, from_date) for c in fields_table.iterrows()]

            data = dict([d.result() for d in as_completed(self.parent.data) if d.result() is not None])
            self.parent.data = data

        if not self.raw:
            data = pd.concat(data.values(), 1).sort_index(axis=1)
        if self.cli_config != {}:
            with pd.HDFStore(self.file_name, 'a') as S:
                S[self.var_code] = data
            print('Field {} fetched and saved in file {}'.format(self.var_code, self.file_name))
        else:
            return data

    def _get(self, f, prog, from_date):
        k = f[0][2]
        try:
            from_date = from_date[f[0][0]] # updating: field exists in old data
        except: pass
        if from_date != from_date: # if Nat
            from_date = pd.Timestamp(f[1]['first']) # updating: earliest record from meta
        try:
            v = from_date.strftime('%Y-%m-%d')
        except:
            v = self.from_date.strftime('%Y-%m-%d') # last resort: global Field.from_date

        try:
            df = self.fetch_raw(f[0][2], v) if self.raw else self.fetch_aggr(f[0][2], v)
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
            self.log.info('fetched {} from {}'.format(f[0][2], f[0][0]))
            prog.update(1)
            return f[0][2], df.sort_index(1)

    def fetch_aggr(self, code, from_date):
        cols = ['ultima_lectura', 'min', 'prom', 'max', 'data_pc']
        params = {
            'fn': 'GetSerieSensor',
            'interv': 'hora',
            'valor_nan': 'nan',
            'user': self.user,
            's_cod': code,
            'fecha_inicio': from_date,
            'fecha_fin': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d'),
        }
        for trial in range(self.parent.trials):
            r = requests.get(self.parent.url, params=params)
            if not r.ok:
                continue
            self.log.debug(r.text)
            reader = _Reader(r.text)
            try:
                d = pd.read_csv(reader, index_col=0, parse_dates=True, usecols=cols)
                reader.close()
            except:
                raise FetchError(r.url)
            else:
                return d.astype(float) # important

        raise TrialsExhaustedError()

    def fetch_raw(self, code, from_date):
        params = {'fi': from_date,
                  'ff': (datetime.utcnow() - timedelta(hours=4)).strftime('%Y-%m-%d'),
                  's_cod': code}
        for trial in range(self.parent.trials):
            r = requests.get(self.raw_url, params=params)
            if not r.ok:
                continue
            self.debug(r.text)
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

        raise TrialsExhaustedError()


class Meta(Application):
    field = Dict().tag(config = True)
    field_index = Instance(slice, (0, 2))
    field_data = Instance(slice, (2, 6))
    station = Dict().tag(config = True)
    file_name = Unicode().tag(config = True)
    get_fields = Bool(True).tag(config = True)

    aliases = Dict({'f': 'Meta.file_name', 'log_level': 'Application.log_level'})
    flags = Dict({'n': ({'Meta': {'get_fields': False}}, "do not fetch field metadata")})

    def start(self):
        self.parent.data = None
        params = {k: v[0] for k, v in self.station.items()}
        for trial in range(self.parent.trials):
            req = requests.get(self.parent.url, params = params)
            if not req.ok:
                continue
            self.log.debug(req.text)
            with StringIO(req.text) as sio:
                try:
                    self.parent.data = [(l[0], l[1: 7]) for l in csv.reader(sio) if l[0][0] != '#']
                except:
                    print('attempt #{}'.format(trial))
                else:
                    break

        if self.parent.data is None:
            raise TrialsExhaustedError()

        cols = [v[1] for k, v in sorted(self.station.items(), key=lambda k: k[0]) if k[0]=='c']
        meta = pd.DataFrame.from_items(
            self.parent.data,
            columns = cols[1:], # 0 is index
            orient='index'
        )
        meta.index.name = cols[0]
        meta.sort_index(inplace=True)

        if self.get_fields:
            with tqdm(total = meta.shape[0]) as prog:
                def get(st):
                    params = {k: v[0] for k, v in self.field.items()}
                    params['e_cod'] = st[0]
                    for trial in range(self.parent.trials):
                        self.log.info(st[1].full)
                        req = requests.get(self.parent.url, params = params)
                        if not req.ok:
                            continue
                        self.log.debug(req.text)
                        with StringIO(req.text) as sio:
                            try:
                                out = [(tuple(np.r_[st[:1], l[self.field_index]]), l[self.field_data])
                                          for l in csv.reader(sio) if l[0][0] != '#']
                            except:
                                print('attempt #{}'.format(trial))
                            else:
                                prog.update(1)
                                return out

                    raise TrialsExhaustedError()

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

class Update(Application):
    overlap = Instance(timedelta, kw={'days': 30}).tag(config = True)
    file_name = Unicode().tag(config = True)
    overwrite = Bool(True).tag(config = True)

    aliases = Dict({'v': 'Field.var_code',
                    'f': 'Field.file_name',
                    'm': 'Meta.file_name',
                    'o': 'Update.file_name',
                    'log_level': 'Application.log_level'})
    flags = Dict({'r': ({'Field': {'raw': True}}, "set raw=True"),
                  'n': ({'Update': {'overwrite': False}}, "Always use Update.file_name to save updated data.")})

    def start(self):
        sta, flds = self.parent.get_meta()
        fields_table = flds.xs(self.config.Field.var_code, 0, 'field', False)
        old_data  = pd.read_hdf(self.config.Field.file_name, self.config.Field.var_code)
        latest = old_data.drop('data_pc', 1, 'aggr').apply(lambda c: c.dropna().index.max()).min(0, level='station')
        # stopped = (sta['last'].astype('datetime64') - datetime.utcnow() + timedelta(hours=4) < - self.overlap)
        from_date = {k: v - self.overlap for k, v in latest.iteritems()}
        # new_codes = fields_table.index.get_level_values('sensor_code').symmetric_difference(old_codes)
        # old_codes = old_data.columns.get_level_values('sensor_code')

        data = self.parent.get_data(None, fields_table, from_date)
        data.update(old_data, overwrite = False)

        with pd.HDFStore(self.config.Meta.file_name) as S:
            S['stations'] = sta
            S['fields'] = flds
            print('Meta data file {} updated.'.format(self.config.Meta.file_name))

        ts = (datetime.utcnow() - timedelta(hours=4))
        if data.shape[1] == old_data.shape[1]:
            try:
                c = Compare(old_data, data)
            except AssertionError:
                self.log.warn('%s: data_pc values between 1 and 99 encountered', ts)
            if len(c.s) > 0:
                self.log.warn('%s: CEAZA.Compare found differing data values', ts)
            elif self.overwrite:
                self.file_name = self.config.Field.file_name
        else:
            self.log.warn('%s: data shape mismatch old %s | new %s', ts, old_data.shape[1], data.shape[1])

        with pd.HDFStore(self.file_name) as N:
            N[self.config.Field.var_code] = data
            print('Updated DataFrame {} saved in file {}.'.format(self.config.Field.var_code, self.file_name))

class CEAZAMet(Application):
    """Class to download data from CEAZAMet webservice. Main reason for having a class is
    to be able to reference the data (CEAZAMet.data) in case something goes wrong at some point.

    """
    url = Unicode().tag(config = True)

    subcommands = Dict({'meta': (Meta, 'get stations and field metadata'),
                        'data': (Field, 'get one field from all stations'),
                        'update': (Update, 'update existing field data')})

    trials = Integer(10)
    max_workers = Integer(16)
    log_file_name = Unicode().tag(config = True) # see initialize()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

    def initialize(self):
        self.parse_command_line()
        if self.log_file_name != '':
            pass # not yet implemented

    def get_meta(self, fields=True):
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
        return app.start()

    def get_data(self, var_code=None, fields_table=None, from_date=None, raw=False):
        """Collect data from CEAZAMet webservice, for one variable type but all stations.

        :param var_code: variable code to be collected (e.g. 'ta_c')
        :param fields_table: pandas.DataFrame with field metadata as constructed by get_stations()
        :param from_date: initial date from which onward to request data
        :param raw: False (default) or True whether raw data should be collected
        :returns: data for one variable and all stations given by var_table
        :rtype: :class:`~pandas.DataFrame` or :obj:`dict` of DataFrames if raw==True

        """
        app = Field(parent=self)
        if var_code is not None: app.var_code = var_code
        if raw is not None: app.raw = raw
        return app.start(fields_table, from_date)

class Compare(object):
    """Compare two CEAZAMet station data DataFrames of one single field (e.g. ta_c).

    :param a: older DataFrame
    :type a: :class:`~pandas.DataFrame`
    :param b: newer DataFrame
    :type b: :class:`~pandas.DataFrame`

    """
    def __init__(self, a, b):
        ft = import_module('functools')
        # just checking that data_pc is all 0 or 100
        assert(not any([(lambda c: np.any((c==c) & (c!=0) & (c!=100)))(c.xs('data_pc', 1, 'aggr')) for c in [a, b]]))

        self.a, self.b = [c.drop('data_pc', 1, 'aggr') for c in [a, b]]
        x = (lambda d: (d == d) & (d != 0))(self.a - self.b)
        # locations where any of the non-data_pc aggr levels are not equal to old data
        self.x = ft.reduce(np.add, [x.xs(a, 1, 'aggr') for a in x.columns.get_level_values('aggr').unique()])

        # stations where not only the last timestamp of old data differs from new data
        z = self.x.apply(lambda c: c.index[c].max() - c.index[c].min(), 0)
        self.s = z[(z == z) & (z > pd.Timedelta(0))]
        print('\nSizes: a {} | b {}\n'.format(a.shape, b.shape))
        if len(self.s) > 0:
            print(self.s)

    def plot(self, stations=None, dt=pd.Timedelta(1, 'D')):
        """Produce overview plot of the differences between the DataFrames.

        :param stations: if too many stations have differences (self.s), plot only a slice
        :type stations: :obj:`slice` or :obj:`None`
        :param dt: timedelta to include on either side of the earliest and latest differing record
        :type dt: :class:`pandas.Timedelta`

        """
        plt = import_module('matplotlib.pyplot')
        sta = self.s.index if stations is None else self.s.index[stations]
        fig, axs = plt.subplots(len(sta), 1)
        for i, ax in zip(sta, axs):
            x, y, z = [j.xs(i[2], 1, 'sensor_code') for j in [self.a, self.b, self.x]]
            d = z.index[z.values.flatten()]
            a, b = d.min() - dt, d.max() + dt
            for idx, c in y.iteritems():
                pl = ax.plot(c.loc[a: b], label=idx[-1])[0]
                ax.plot(x.xs(idx[-1], 1, 'aggr').loc[d], 'x', color=pl.get_color())
                ax.vlines(d, *ax.get_ylim(), color='r', label='_')
        plt.legend()

if __name__ == '__main__':
    app = CEAZAMet()
    app.initialize()
    app.subapp.start()
