#!/usr/bin/env python
"""
CEAZAMet stations webservice
----------------------------

This module contains classes to interact with the CEAZAMet webservice. The two main classes are:

    * :class:`Meta` - for downloading station meta data
    * :class:`Field` - for downloading data corresponding to a particular sensor type

Furthermore, there are some utility classes, e.g. :class:`compare` to compare datasets.

Command-line use
================

Work in progress.

Updating
========

Various classes feature an 'update' method: :meth:`Meta.update`, :meth:`Field.update`.

Logging
=======

There's some rudimentary logging to the file specified as :attr:`Common.log_file`

.. TODO::

    * DEM interpolation in :class:`.Update`
    * updating hourly data (?)
    * work through meta data update

"""
import requests as reqs
import csv, os, sys, re, logging
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from traitlets.config import Application, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode, Instance, Dict, Integer, Bool
from importlib import import_module
from tqdm import tqdm
from helpers import config
from functools import reduce

class compare(config.CEAZAMet):
    """Helper to spot differences between datasets. Currently mainly focused on :obj:`dicts <dict>` with 'raw' :class:`DataFrames <pandas.DataFrame>`.

    .. attribute:: overlap

        Test

    .. attribute:: indexes

        Test

    """
    def __init__(self, **kwargs):
        if kwargs != {}:
            self.load(**kwargs)

    def load(self, var_code, raw=True):
        self.var_code = var_code
        with pd.HDFStore({True: self.raw_data, False: self.hourly_data}[raw]) as S:
            node = S.get_node('raw/{}'.format(var_code))
            self.data = {k: S[v._v_pathname] for k, v in node._v_children.items()}

    @property
    def last(self):
        return pd.Series({k: v.dropna().index.max() for k, v in self.data.items()}, name='data')

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            m = self.fields.xs(self.var_code, 0, level='field').set_index('sensor_code')
            self._meta = pd.Series(m['last'], name='last')
        return self._meta

    @property
    def fields(self):
        if not hasattr(self, '_flds'):
            self._flds = pd.read_hdf(self.meta_data, 'fields')
        return self._flds

    def compare_dicts(self, a, b):
        self.indexes, self.overlap = {}, pd.DataFrame()
        keys = set(a.keys()).union(b.keys())
        df = pd.DataFrame(np.zeros((len(keys), 2)) * np.nan, index=keys, columns=['a', 'b'])
        for k in keys:
            try:
                assert a[k] is not None
                assert b[k] is not None
            except:
                df.loc[k, 'a'] = 1 if a.get(k, None) is not None else 0
                df.loc[k, 'b'] = 1 if b.get(k, None) is not None else 0
            else:
                e, n = self.compare_dataframes(a[k], b[k])
                self.overlap = self.overlap.append(pd.Series(n, name=k))
                if e == {}:
                    df.drop(k, inplace=True)
                else:
                    self.indexes[k] = e
        df = pd.concat((df, self.last, self.meta), 1)
        return df[df.a.notnull()]

    @staticmethod
    def compare_dataframes(a, b):
        idx = a.index.intersection(b.index)
        c = (a.loc[idx].fillna(-9999) == b.loc[idx].fillna(-9999))
        d = {k: np.where(v==False)[0] for k, v in c.iteritems() if not v.all()}
        return d, {'start':idx.min(), 'end':idx.max()}

class requests:
    url = ''
    @classmethod
    def get(cls, *args, **kwargs):
        r = reqs.get(*args, **kwargs)
        r.raise_for_status()
        cls.url = r.url
        return r

class NoNewStationError(Exception):
    pass

class TrialsExhaustedError(Exception):
    pass

class ServerMemoryError(Exception):
    pass

class FieldsUpToDate(Exception):
    pass

class base_app(Application):
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    def __init__(self, *args, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound: cfd = Config(config)
        super().__init__(config=cfg, **kwargs)

class Common(config.CEAZAMet):
    """Common functionality and config values."""
    max_workers = 10
    earliest_date = datetime(2003, 1, 1)
    timedelta = pd.Timedelta(-4, 'h')
    update_overlap = pd.Timedelta(1, 'd')
    update_drop = pd.Timedelta(180, 'd')
    memory_err =  re.compile('allowed memory', re.IGNORECASE)

    def __init__(self):
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG)

    def dates(self, from_date, to_date):
        try:
            from_str = from_date.strftime('%Y-%m-%d')
        except:
            from_date = self.earliest_date
            from_str = from_date.strftime('%Y-%m-%d')
        try:
            to_str = to_date.strftime('%Y-%m-%d')
        except:
            to_date = datetime.utcnow() + self.timedelta
            to_str = to_date.strftime('%Y-%m-%d')
        return from_date, to_date, from_str, to_str

    def read(self, url, params, cols, n_trials=10, prog=None, **kwargs):
        for trial in range(n_trials):
            req = requests.get(url, params=params)
            logging.debug('fetching {}'.format(req.url))
            self.memory_check(req.text)
            # self.log.debug(req.text)
            with StringIO(req.text) as sio:
                try:
                    df = pd.read_csv(sio, index_col=0, comment='#', header=None).dropna(1, 'all')
                    df.columns = cols[1:]
                    df.index.name = cols[0]
                    df.sort_index(inplace=True)
                except:
                    logging.warning('attempt #{}'.format(trial))
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

    def get_all(self, var_code, fields_table=None, from_date=None, to_date=None, raw=False):
        self.var_code = var_code
        if fields_table is None:
            logging.info('Using field table from file {}'.format(self.meta_data))
            fields_table = pd.read_hdf(self.meta_data, 'fields')
        table = fields_table.xs(var_code, 0, 'field', False)

        # NOTE: I'm not sure if tqdm is thread-safe
        with tqdm(total = table.shape[0]) as prog:
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                self.data = [exe.submit(self._get, r, prog, from_date=from_date, to_date=to_date, raw=raw)
                             for r in table.iterrows()]
            data = dict([d.result() for d in as_completed(self.data)])
        if raw:
            self.no_data = [k for k, v in data.items() if v is None]
            last = table.set_index('sensor_code').loc[self.no_data]['last'].astype('datetime64')
            logging.warning('No data retrieved from {}'.format(self.no_data))
            logging.warning('Last data within {} of now: {}'.format(
                self.update_drop, last[datetime.utcnow() - last < self.update_drop]
            ))
            self.data = {k: v for k, v in data.items() if v is not None}
        else:
            self.data = pd.concat(data.values(), 1, sort=True).sort_index(axis=1)

    def _get(self, fields_row, prog=None, from_date=None, **kwargs):
        (station, field), row = fields_row
        try:
            # updating: field exists in old data
            from_date = row['update'] - self.update_overlap
        except: pass
        if from_date is None or from_date != from_date: # if NaT
            try: from_date = pd.Timestamp(row['first']) # updating: earliest record from meta
            except: pass
        if hasattr(self, 'update_times'):
            self.update_times.loc[row['sensor_code'], 'from_date'] = from_date

        df = self.fetch(row['sensor_code'], from_date, **kwargs)
        if df is None:
            return row['sensor_code'], None
        df.index = pd.DatetimeIndex(df.index)

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

    def fetch(self, code, from_date=None, to_date=None, raw=False):
        from_date, to_date, from_str, to_str = self.dates(from_date, to_date)
        params = {
            False: {
                'fn': 'GetSerieSensor',
                'interv': 'hora',
                'valor_nan': 'nan',
                'user': self.user,
                's_cod': code,
                'fecha_inicio': from_str,
                'fecha_fin': to_str
            },
            True: {
                'fi': from_str,
                'ff': to_str,
                's_cod': code
            }
        }[raw]
        url = {True: self.raw_url, False: self.url}[raw]
        cols = self.cols[:-1] if raw else self.cols
        try:
            return self.read(url, params, cols).set_index('datetime')

        # for too long data series, the server throws an error, see Common.memory_check
        except ServerMemoryError:
            dt = (to_date - from_date) / 2
            # break if the interval-halving runs away
            assert dt > timedelta(days=365), code
            mid = from_date + dt
            dfs = [d for d in
                   (self.fetch(code, from_date, mid, raw), self.fetch(code, mid, to_date, raw))
                   if d is not None]
            try: return dfs[0].combine_first(dfs[1])
            except:
                try: return dfs[0]
                except: return None
        except:
            logging.info('Possibly no data for {}: from {} to {}'.format(code, from_date, to_date))
            return None


    def update(self, var_code, raw=True, **kwargs):
        assert raw, "Currently only raw updating is implemented"
        self.var_code = var_code
        with pd.HDFStore(self.meta_data) as M:
            flds = M['fields'].xs(var_code, 0, 'field', drop_level=False)
            last = flds[['sensor_code', 'last']].set_index('sensor_code').squeeze()

        filename = {
            True: self.raw_data,
            False: self.hourly_data
        }[raw]

        with pd.HDFStore(filename, 'a') as S:
            node = S.get_node('raw/{}'.format(var_code))
            data = {k: S[v._v_pathname] for k, v in node._v_children.items()}
            update = pd.Series({k: v.dropna().index.max() for k, v in data.items()}).astype('datetime64')
            table = flds.merge(
                update.to_frame('update'),
                how='left',
                left_on='sensor_code',
                right_index=True
            )
            delta = flds.merge(
                (datetime.utcnow() - update.combine_first(last)).to_frame('delta'),
                how='left',
                left_on='sensor_code',
                right_index=True
            )['delta']
            try:
                # this should avoid problems with NaNs in delta, but will probably break if they're all NaNs
                assert delta.min() > self.update_overlap
            except:
                raise FieldsUpToDate()
            self.get_all(var_code, fields_table=table[delta < self.update_drop], raw=raw, **kwargs)
            start = pd.Series({k: v.dropna().index.min() for k, v in self.data.items() if v is not None})
            # this is a diagnostic for checks
            self.update_times = pd.concat((last, update, start), 1, keys=['last', 'update', 'start'])
            d = {}
            for k in set(data.keys()).union(self.data.keys()):
                if k in data and data[k] is not None:
                    d[k] = data[k]
                    if k in self.data and self.data[k] is not None:
                        d[k] = self.data[k].combine_first(d[k])
                elif self.data[k] is not None:
                    d[k] = self.data[k]
            for k, v in d.items():
                S['/'.join(('raw', var_code, k))] = v

    def store_raw(self, filename):
        with pd.HDFStore(filename, 'w') as S:
            for k, v in self.data.items():
                if v is not None and v.shape[0] > 0:
                    S['raw/{}/{}'.format(self.var_code, k)] = v

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
        super().__init__()
        self.__dict__.update({'_'+k: v for k, v in kwargs.items()})

    @property
    def stations(self):
        if not hasattr(self, '_stations'):
            params, cols = zip(*self.station)
            params = {'c{:d}'.format(i): v for i, v in enumerate(params)}
            self._stations = self.read(self.url, params, cols, fn='GetListaEstaciones', p_cod='ceazamet')
        return self._stations

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            with tqdm(total = self.stations.shape[0]) as prog:
                def get(st):
                    params, cols = zip(*self.field)
                    params = {'c{:d}'.format(i): v for i, v in enumerate(params)}
                    return self.read(
                        self.url, params, cols,
                        fn='GetListaSensores', e_cod=st, p_cod='ceazamet', prog=prog)

                with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                    field_meta = exe.map(get, self.stations.index)

                self._fields = pd.concat(field_meta, 0, keys=self.stations.index)

        return self._fields

    def store(self, **kwargs):
        with pd.HDFStore(self.meta_data, mode='a') as S:
            change_key = datetime.now().strftime('%Y%m%d')
            for k in ['stations', 'fields']:
                S[os.path.join(change_key, k)] = S[k]
                S[k] = kwargs.get(k, getattr(self, k))

        print('CEAZAMet station metadata saved in file {}.'.format(self.meta_data))
        print('Old contents moved to node {}.'.format(change_key))

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

    def update(self, **kwargs):
        sta = kwargs.get('stations', pd.read_hdf(self.meta_data, 'stations'))
        flds = kwargs.get('fields', pd.read_hdf(self.meta_data, 'fields'))

        flds['elev'] = flds['elev'].astype(float)
        flds.reset_index('sensor_code', inplace=True)

        self._stations = self.stations.combine_first(sta)
        self._fields = self.update_fields(flds)

    def update_fields(self, flds):
        outer = self.fields.merge(flds, how='outer')
        left = self.fields.merge(flds, how='left')
        for s in [k for k, c in np.vstack(np.unique(outer.sensor_code, return_counts=True)).T if c>1]:
            o = outer[outer.sensor_code==s]
            l = left[left.sensor_code==s].copy()
            try:
                # check if everything other than first, last is the same
                assert (lambda x: np.all(x.iloc[0]==x.iloc[1]))(o.drop(['first', 'last'], 1))
            except:
                # some soil temp elevations were mislabeled as positive in older versions of the db
                assert o['elev'].sum() == 0, o['elev']
                l['elev'] = o['elev'].dropna().min()

            # use the widest date range for first, last
            try: l['first'] = o['first'].dropna().min()
            except: pass
            try: l['last'] = o['last'].dropna().min()
            except: pass
            outer.drop(o.index, 0, inplace=True)
            outer = outer.append(l)

        cc = pd.concat((self.fields, flds), 0, sort=True)
        outer.index = reduce(lambda a, b: a.append(b),
                             [cc.index[cc['sensor_code']==c].unique() for c in outer['sensor_code']])
        return outer.sort_index()


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
        # just checking that data_pc is all 0 or 100
        assert(not any([(lambda c: np.any((c==c) & (c!=0) & (c!=100)))(c.xs('data_pc', 1, 'aggr')) for c in [a, b]]))

        self.a, self.b = [c.drop('data_pc', 1, level='aggr').rename({'prom': 'ave'}, axis=1, level='aggr')
                          for c in [a, b]]
        x = (lambda d: (d == d) & (d != 0))(self.a - self.b)
        # locations where any of the non-data_pc aggr levels are not equal to old data
        self.x = reduce(np.add, [x.xs(i, 1, 'aggr') for i in x.columns.get_level_values('aggr').unique()])

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
            ax.set_ylabel(i[0])
        plt.legend()

if __name__ == '__main__':
    f = Field()
    f.update('ta_c')
