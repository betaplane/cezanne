#!/usr/bin/env python
"""
CEAZAMet stations webservice
----------------------------

This module can be imported and used as a standalone command-line app. The use as module is documented in the class :class:`CEAZAMet`.

Command-line use
================

There are three subcommands, ``meta``, ``data`` and ``update``, with the first two corresponding to the methods :meth:`get_meta` and :meth:`get_data`. They can be supplied with command-line arguments in the `IPython  <https://ipython.readthedocs.io/en/stable/config/index.html>`_ / `Jupyter <https://jupyter.readthedocs.io/en/latest/projects/config.html>`_ config style. This means that help is also available in the same style, e.g.::

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

Updating
========

When the ``update`` subcommand to the command-line program is used, only 'new' data will be downloaded from the stations. 'New' is defined by extracting the table named :attr:`.Field.var_code` from the file given in :attr:`.Field.file_name` and taking the timestamp from the last record in this table. ``Update`` will download data that overlaps with the existing one by the amount of time specified as :attr:`.Update.overlap` and compare it to the existing data by using :class:`.Compare` (which prints a summary dataframe with differences if such are found). If :attr:`.Update.overwrite` is ``True`` **and** no differences are found, the existing data is replaced, otherwise it is written to the file specified as :attr:`.Update.file_name`.

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

    * DEM interpolation in :class:`.Update`
    * save raw data (filename = field_code, tablename = station_code)
    * rework printed info (log?)

"""
import requests, csv, os, sys, re
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from traitlets.config import Application, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode, Instance, Dict, Integer, Bool
from importlib import import_module
from tqdm import tqdm
from helpers import config
from functools import partial


class NoNewStationError(Exception):
    pass

class TrialsExhaustedError(Exception):
    pass

class ServerMemoryError(Exception):
    pass

class base_app(Application):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    def __init__(self, *args, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound: cfd = Config(config)
        super().__init__(config=cfg, **kwargs)

class Common:
    max_workers = 10
    earliest_date = datetime(2003, 1, 1)
    timedelta = timedelta(hours=-4)
    memory_err =  re.compile('allowed memory', re.IGNORECASE)

    def from_date(self, date):
        try:
            return date.strftime('%Y-%m-%d')
        except:
            return self.earliest_date.strftime('%Y-%m-%d')

    def to_date(self, to_date=None):
        try:
            return to_date.strftime('%Y-%m-%d')
        except:
            return (datetime.utcnow() + self.timedelta).strftime('%Y-%m-%d')

    def read(self, url, params, cols, n_trials=10, prog=None, **kwargs):
        params.update(kwargs)
        for trial in range(n_trials):
            req = requests.get(url, params=params)
            req.raise_for_status()
            self.memory_check(req.text)
            # self.log.debug(req.text)
            with StringIO(req.text) as sio:
                try:
                    df = pd.read_csv(sio, index_col=0, comment='#', header=None).dropna(1, 'all')
                    df.columns = cols[1:]
                    df.index.name = cols[0]
                    df.sort_index(inplace=True)
                except:
                    print('attempt #{}'.format(trial))
                else:
                    if prog is not None:
                        prog.update(1)
                return df.sort_index()

        raise TrialsExhaustedError()

    @classmethod
    def memory_check(cls, s):
        try:
            assert not cls.memory_err.search(s)
        except:
            raise ServerMemoryError()


class Field(Common):
    cols = ['s_cod', 'datetime', 'min', 'ave', 'max', 'data_pc']

    def start(self, var_code, fields_table=None, from_date=None, raw=False):
        if fields_table is None:
            print('Using field table from file {}'.format(config.Meta.file_name))
            fields_table = pd.read_hdf(config.Meta.file_name, 'fields')
        table = fields_table.xs(var_code, 0, 'field', False)

        # NOTE: I'm not sure if tqdm is thread-safe
        with tqdm(total = table.shape[0]) as prog:
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                self.data = exe.map(partial(self._get, prog=prog, from_date=from_date, raw=raw), table.iterrows())

            data = dict([d for d in self.data if d is not None])

        self.data = data if raw else pd.concat(data.values(), 1).sort_index(axis=1)


        # this tests whether the class is called as a command-line app, in which case the data is saved to file
        # EXCEPT if the command line app is "Update"
        if False: # I'm moving away from traitlets
            with pd.HDFStore(self.file_name, 'a') as S:
                if self.raw:
                    for k, v in data.items():
                        st = fields_table.xs(k, 0, 'sensor_code').index.item()[0]
                        try:
                            S[st] = v.update(S[st])
                        except:
                            S[st] = v
                else:
                    S[self.var_code] = data
            print('Field {} fetched and saved in file {}'.format(self.var_code, self.file_name))

    def _get(self, fields_row, prog=None, from_date=None, raw=False):
        (station, field), row = fields_row
        try:
            from_date = from_date[station] # updating: field exists in old data
        except: pass
        if from_date is None or from_date != from_date: # if Nat
            from_date = pd.Timestamp(row['first']) # updating: earliest record from meta

        df = self.fetch(row['sensor_code'], from_date, raw=raw)
        i = df.columns.size
        df.columns = pd.MultiIndex.from_arrays(
            np.r_[
                np.repeat([station, field, row['sensor_code'], row['elev']], i),
                df.columns
            ].reshape((-1, i)),
            names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
        )
        # self.log.info('fetched {} from {}'.format(f[0][2], f[0][0]))
        if prog is not None:
            prog.update(1)
        return row['sensor_code'], df.sort_index(1)

    def fetch(self, code, from_date, to_date=None, raw=False):
        params = {
            False: {
                'fn': 'GetSerieSensor',
                'interv': 'hora',
                'valor_nan': 'nan',
                'user': config.CEAZAMet.user,
                's_cod': code,
                'fecha_inicio': self.from_date(from_date),
                'fecha_fin': self.to_date(to_date)
            },
            True: {
                'fi': self.from_date(from_date),
                'ff': self.to_date(to_date),
                's_cod': code
            }
        }[raw]
        print(params)
        url = {True: config.CEAZAMet.raw_url, False: config.CEAZAMet.url}[raw]
        cols = self.cols[:-1] if raw else self.cols
        try:
            return self.read(url, params, cols).set_index('datetime')

        # for too long data series, the server throws an error, see Common.memory_check
        except ServerMemoryError:
            from_date = from_date if isinstance(from_date, datetime) else self.earliest_date
            to_date = datetime.utcnow() + self.timedelta if to_date is None else to_date
            dt = (to_date - from_date) / 2
            assert dt > timedelta(days=365)
            mid = from_date + dt
            df1 = self.fetch(code, from_date, mid, raw)
            df2 = self.fetch(code, mid, to_date, raw)
            return pd.concat((df1, df2)).dropna('all')
        except:
            print('Possibly no data for {}: from {} to {}'.format(code, from_date, to_date))
            return None


class Meta(Common):
    field = [
        ('tm_cod', 'field'),
        ('s_cod', 'sensor_code'),
        ('tf_nombre', 'full'),
        ('um_notacion', 'unit'),
        ('s_altura', 'elev'),
        ('s_primera_lectura', 'first'),
        ('s_ultima_lectura', 'last')
    ]

    station = [
        ('e_cod', 'station'),
        ('e_nombre', 'full'),
        ('e_lon', 'lon'),
        ('e_lat', 'lat'),
        ('e_altitud', 'elev'),
        ('e_primera_lectura', 'first'),
        ('e_ultima_lectura', 'last')
    ]
    def __init__(self, **kwargs):
        self.__dict__.update({'_'+k: v for k, v in kwargs.items()})

    @property
    def stations(self):
        if not hasattr(self, '_stations'):
            params, cols = zip(*self.station)
            params = {'c{:d}'.format(i): v for i, v in enumerate(params)}
            self._stations = self.read(
                config.CEAZAMet.url, params, cols, fn='GetListaEstaciones', p_cod='ceazamet')

        return self._stations

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            with tqdm(total = self.stations.shape[0]) as prog:
                def get(st):
                    params, cols = zip(*self.field)
                    params = {'c{:d}'.format(i): v for i, v in enumerate(params)}
                    return self.read(
                        config.CEAZAMet.url, params, cols,
                        fn='GetListaSensores', e_cod=st, p_cod='ceazamet', prog=prog)

                with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                    field_meta = exe.map(get, self.stations.index)

                self._fields = pd.concat(field_meta, 0, keys=self.stations.index)

        return self._fields

    def store(self):
        with pd.HDFStore(config.Meta.file_name, mode='w') as S:
            if hasattr(self, 'meta'):
                S['stations'] = self.meta
            if hasattr(self, 'fields'):
                S['fields'] = self.fields
        print("CEAZAMet station metadata saved in file {}.".format(self.file_name))

    @staticmethod
    def model_elevation(meta, dem, var_name, column_name):
        """Interpolate a gridded elevation dataset (e.g. the DEM used by WRF internally) to station locations and append the model elevations to the metadata DataFrame.

        :param meta: the metadata DataFrame
        :type meta: :class:`~pandas.DataFrame`
        :param dem: the elevation dataset (needs to be the whole dataset for the interpolator to get the metadata)
        :type dem: :class:`~xarray.Dataset`
        :param var_name: name of the elevation variable in the ``dem`` Dataset
        :param column_name: name the appended elevation column should be given
        :returns: new metadata DataFrame with DEM elevation data appended
        :rtype: :class:`~pandas.DataFrame`

        """
        ip = import_module('interpolate', 'data')
        intp = ip.GridInterpolator(dem)
        z = intp.xarray(dem[var_name]).squeeze()
        return pd.concat((meta, pd.DataFrame(z, index=z.stations, columns=[column_name])), 1)

class Update(base_app):
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

class CEAZAMet(base_app):
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
