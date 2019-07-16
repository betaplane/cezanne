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

Currently not implemented. Command-line usage is enabled in :mod:`WRF` and this module is called from the function :func:`WRF.update_ceazamet`.

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


logger = logging.getLogger(__name__)
# this is supposed to get executed only once (in particular not if 'reload' is used) -
# otherwise each logged line is added to the file as many times as identical file handlers have been added
if len(logger.handlers) == 0:
    fh = logging.FileHandler(config.CEAZAMet.log_file)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

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


class Common(config.CEAZAMet):
    """Common functionality and config values."""
    max_workers = 10
    earliest_date = datetime(2003, 1, 1)
    timedelta = pd.Timedelta(-4, 'h')
    update_overlap = pd.Timedelta(1, 'd')
    update_drop = pd.Timedelta(180, 'd')
    memory_err =  re.compile('allowed memory', re.IGNORECASE)

    def __init__(self):
        s = '{} @ {}'.format(self.__class__.__name__, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('\n{}\n{}'.format(s, ''.join(['=' for _ in range(len(s))])))

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
            params.update(kwargs)
            req = requests.get(url, params=params)
            logger.debug('fetching {}'.format(req.url))
            self.memory_check(req.text)
            # self.log.debug(req.text)
            with StringIO(req.text) as sio:
                try:
                    df = pd.read_csv(sio, index_col=0, comment='#', header=None).dropna(1, 'all')
                    df.columns = cols[1:]
                    df.index.name = cols[0]
                    df.sort_index(inplace=True)
                except:
                    logger.warning('attempt #{}'.format(trial))
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
        if fields_table is None:
            logger.info('Using field table from file {}'.format(self.meta_data))
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
            logger.warning('No data retrieved from {}'.format(self.no_data))
            logger.warning('Last data within {} of now: {}'.format(
                self.update_drop, last[datetime.utcnow() - last < self.update_drop]
            ))
            self.data = {k: v for k, v in data.items() if v is not None}
        else:
            self.data = pd.concat(data.values(), 1, sort=True).sort_index(axis=1)

    def _get(self, fields_row, prog=None, from_date=None, **kwargs):
        (station, field), row = fields_row
        # NOTE: 'update' in row has priority over from_date, *but* from_date is the *earliest*
        try:
            # updating: field exists in old data
            d = row['update'] - self.update_overlap
            from_date = d if from_date is None else (maxd, from_date)
        except: pass
        if from_date is None or from_date != from_date: # if NaT
            try: from_date = pd.Timestamp(row['first']) # updating: earliest record from meta
            except: pass
        if hasattr(self, 'update_times'):
            self.update_times.loc[row['sensor_code'], 'from_date'] = from_date

        df = self.fetch(row['sensor_code'], from_date, **kwargs)
        if prog is not None:
            prog.update(1)
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
            logger.info('Possibly no data for {}: from {} to {}'.format(code, from_date, to_date))
            return None

    def update(self, var_code, raw=True, **kwargs):
        assert raw, "Currently only raw updating is implemented"
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
        now = kwargs.get('to_date', None)
        if now is None:
            now = datetime.utcnow()
        delta = flds.merge(
            (now - update.combine_first(last)).to_frame('delta'),
            how='left',
            left_on='sensor_code',
            right_index=True
        )['delta']
        try:
            # this should avoid problems with NaNs in delta, but will probably break if they're all NaNs
            assert delta.min() > (self.update_overlap / 2)
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
            v.to_hdf(filename, key='/'.join(('raw', var_code, k)), mode='a', format='table')

    def store_raw(self, filename=None):
        if filename is None:
            filename = self.raw_data
        for k, v in self.data.items():
            if v is not None and v.shape[0] > 0:
                var_code = v.columns.get_level_values('field').unique().item()
                v.to_hdf(filename, key='raw/{}/{}'.format(var_code, k), mode='a', format='table')

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

    @property
    def stations(self):
        if not hasattr(self, '_stations'):
            logger.info('\nstations\n********')
            params, cols = zip(*self.station)
            params = {'c{:d}'.format(i): v for i, v in enumerate(params)}
            self._stations = self.read(self.url, params, cols, fn='GetListaEstaciones', p_cod='ceazamet')
        return self._stations

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            logger.info('\nfields\n******')
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

        logger.info("\nstore\n*****")
        logger.info('CEAZAMet station metadata saved in file {}.'.format(self.meta_data))
        logger.info('Old contents moved to node {}.'.format(change_key))

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
        try:
            flds.reset_index('sensor_code', inplace=True)
        except: pass

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


class Tools:
    class HDF:
        @staticmethod
        def dict(var_code, filename=config.CEAZAMet.raw_data):
            with pd.HDFStore(filename) as S:
                node = S.get_node('raw/{}'.format(var_code))
                return {k: S[v._v_pathname] for k, v in node._v_children.items()}

        @staticmethod
        def variables(filename=config.CEAZAMet.raw_data):
            with pd.HDFStore(filename) as S:
                node = S.get_node('raw')
                return [v for v in node._v_children.keys() if v!='binned']

        @classmethod
        def dates(cls, filename=config.CEAZAMet.raw_data):
            variables = cls.variables(filename)
            mm = lambda idx: (idx.min(), idx.max())
            return pd.concat([
                pd.DataFrame({k: mm(df.index) for k, df in cls.dict(v, filename).items()},
                             index=('start', 'end')).T
                for v in variables], 1, keys=variables)

        @classmethod
        def minutes(cls, filename=config.CEAZAMet.raw_data):
            variables = cls.variables(filename)
            return {v: {k: tuple(sorted(df.index.minute.unique()))
                        for k, df in cls.dict(v, filename).items()} for v in variables}

        @staticmethod
        def sensors(sensor_codes):
            flds = pd.read_hdf(config.CEAZAMet.meta_data, 'fields').reset_index().set_index('sensor_code')
            return flds.loc[sensor_codes]

        @staticmethod
        def pivot(df):
            idx = pd.DatetimeIndex(df.index.date)+pd.TimedeltaIndex(df.index.hour, 'h')
            return df.assign(minute=df.index.minute).assign(idx=idx).pivot(index='idx', columns='minute')

        @staticmethod
        def interval_df1(df):
            sta = df.columns.get_level_values('station').unique().item()
            sensor = df.columns.get_level_values('sensor_code').unique().item()
            idx = pd.DatetimeIndex(df.index.date)+pd.TimedeltaIndex(df.index.hour, 'h')
            try:
                d = df.assign(minute=df.index.minute).assign(idx=idx).pivot(index='idx', columns='minute')
                s = d.notnull().sum(1)
                counts = np.vstack(sorted(np.vstack(np.unique(s, return_counts=True)).T,key=lambda x:x[1]))
            except:
                return (sta, sensor), None, None, None, None, "pivot"
            else:
                try:
                    switch = s.index[s==counts[-1, 0]].min()
                    iv2, iv1 = sorted(60 / counts[-2:, 0] * df.shape[1])
                    n = sorted(counts[-2:, 0])
                except:
                    return (sta, sensor), None, None, None, None, "switch"
                else:
                    try:
                        assert s.index[s==n[0]].values.astype(float).mean() < s.index[s==n[1]].values.astype(float).mean()
                    except:
                        return (sta, sensor), iv1, iv2, switch, None, "assert"
                    else:
                        try:
                            m = d.columns.get_level_values('minute')
                            ix1, ix2 = [m.get_indexer_for(np.arange(n[0]/df.shape[1]) * i) for i in (iv1, iv2)]
                            jx = list(set(ix2)-set(ix1))
                            dc = d.loc[:switch]
                            check = dc.index[dc.iloc[:, jx].isnull().sum(1)==len(jx)].max()
                            try:
                                cx = df.loc[check:switch].iloc[1:-1]
                                assert cx.index.hour.unique().item() == switch.hour - 1
                                switch = cx.index.min()
                                cx = df.loc[check:switch].iloc[1:-1]
                            except: pass
                            return (sta, sensor), iv1, iv2, switch, check, cx.shape[0]
                        except:
                            return (sta, sensor), iv1, iv2, switch, None, None

        @classmethod
        def interval_df2(cls, df):
            def mode(idx):
                d = np.diff(idx).astype('timedelta64[m]').astype(int)
                return sorted(np.vstack(np.unique(d, return_counts=True)).T, key=lambda i:i[1])[-1][0].item()
            m = [mode(df.index[slice(*s)]) for s in [(100,), (-100, None), (2,), (-2, None)]]
            if m[:2]!=m[2:]: return (m,)
            else:
               if m[0]==m[1]:
                   o = cls.interval_outliers(df, m[:1])
                   return m[:1], o
               else:
                   o = cls.interval_outliers(df, m[:2])
                   t = cls.interval_change(df)
                   return m[:2], o, t

        @staticmethod
        def interval_change(df):
            from sklearn.tree import DecisionTreeClassifier
            tr = DecisionTreeClassifier(max_leaf_nodes=2)
            t = df.index.values[1:].astype(float).reshape((-1, 1))
            a = tr.fit(t, np.diff(df.index).astype('timedelta64[m]').astype(int)).predict(t)
            return df.index[1:][a==a[0]].max()

        @staticmethod
        def interval_outliers(df, intervals):
            m = set(df.index.minute)
            x = set(np.hstack(np.arange(60/iv) * iv for iv in intervals))
            return m-x


        @classmethod
        def interval_dfs(cls, var_code, filename=config.CEAZAMet.raw_data):
            flds = pd.read_hdf(config.CEAZAMet.meta_data, 'fields')
            flds = flds.reset_index(level='field').set_index('sensor_code', append=True)
            d = cls.dict(var_code, filename)
            df = pd.DataFrame(cls.interval_df1(df) for df in d.values())
            df.index = pd.MultiIndex.from_tuples(df[0], names=['station', 'sensor_code'])
            df.drop(0, 1, inplace=True)
            df.columns = ['iv1', 'iv2', 'switch', 'check', 'gaplength']
            return df.join(flds)

        @classmethod
        def intervals(cls, filename=config.CEAZAMet.raw_data):
            return pd.concat([cls.interval_dfs(v, filename) for v in cls.variables(filename)]).sort_index()

        @staticmethod
        def check_change(df, ts):
            loc = df.index.get_loc(ts)
            return df.iloc[loc-20:loc+20]


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
    pass
