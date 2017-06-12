#!/usr/bin/env python
from os.path import join as jo
from os import walk
import pandas as pd
from datetime import timedelta
from glob import glob
from configparser import ConfigParser
from CEAZAMet import Downloader, NoNewStationError
# from WRF import WRFOUT
import WRF


class Data(object):
    def __init__(self, config='data.cfg'):
        self.conf = ConfigParser()
        self.conf.read(config)
        self._data = {}
        base = self.conf['base']['path']
        self._paths = {k: jo(base, v) for k, v in self.conf['paths'].items()}
        self.open('_sta', 'stations.h5')

    @property
    def sta(self):
        return self._sta['stations']

    @sta.setter
    def sta(self, value):
        self._sta['stations'] = value

    @property
    def flds(self):
        return self._sta['fields']

    @flds.setter
    def flds(self, value):
        self._sta['fields'] = value

    def open(self, key, name):
        def gl(s):
            return [f for g in [glob(jo(p, '**', s), recursive=True)
                              for p in self._paths.values()] for f in g]
        fl = gl(name)
        if len(fl) != 1:
            fl = gl('*{}*'.format(name))
        if len(fl) != 1:
            print('{} files found'.format(len(fl)))
        else:
            try:
                self._data[key] = pd.HDFStore(fl[0])
            except:
                print('Not a HDF5 file.')

    def append(self, key, var, sta=None, dt=-4):
        d = self._data[key]
        df = d[var]
        m = d['meta'][var].to_dict()
        typ = m.pop('type')
        if typ == 'netcdf':
            D = self._append_netcdf(df, var, m, sta, dt)
        elif typ == 'ceazamet' or type == 'ceazaraw':
            D = self._append_ceazamet(df, var, typ)
        return D
        # self._data[key][var] = D

    @staticmethod
    def _dt(df, field=None, sensor=False):
        if field is not None:
            df = df.xs(field, 1, 'field')
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        if sensor:
            return t, df.columns.get_level_values('sensor_code').unique()
        else:
            return t

    def update_netcdf(self, df, var, sta=None, dt=-4, **meta):
        t = self._dt(df)
        h = t.replace(hour=meta['hour'])
        if h > t - timedelta(hours=dt-1):
            h = h - timedelta(days=1)
        self.OUT = WRF.OUT(paths=self.conf['wrf'].values(), from_date=h, **meta)
        n = self.OUT.netcdf([var], sta=sta)
        return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)

    def update_ceazamet(self, d, var, typ):
        raw = True if typ=='ceazaraw' else False

        # make sure we have enough overlap - webservice for averaged data always reports until 23:00h
        # even if there is not data yet (i.e. NaNs if time is < 23:00h)
        dt = timedelta(days=2)
        if raw:
            t, o = zip(*[self._dt(d[k], var, True) for k in d.keys()])
            t = max(t) - dt
            o = [x for y in o for x in y]
        else:
            t, o = self._dt(d, var, True) - dt

        self.down = Downloader()
        try:
            self.sta, self.flds = self.down.get_stations()
        except NoNewStationError:
            pass

        idx = pd.IndexSlice
        n = set(self.flds.index.get_level_values('sensor_code')) - set(o) # new fields
        if len(n):
            f = self.flds.loc[idx[:, :, list(n)], :]
            b = self.down.get_field(var, f, raw=raw)
            if not raw:
                d = pd.concat((d, b), 1)

        # old fields
        f = self.flds.loc[idx[:, :, o], :]
        a = self.down.get_field(var, f, from_date=t.date(), raw=raw)

        dt = timedelta(seconds=1) # to get the indexing right
        if raw:
            for k, v in a.items():
                d[k] = pd.concat((d[k][:v.index[0] - dt], v), 0)
            if len(n):
                for k, v in b.items():
                    d[k] = v
        else:
            d = pd.concat((d[:a.index[0] - dt], a), 0)
        return d

    def close(self, key=None):
        if key is None:
            sta = self._data.pop('_sta')
            while self._data:
                k, v = self._data.popitem()
                v.close()
            self._data['_sta'] = sta
        else:
            v = self._data.pop(key)
            v.close()

    def ls(self):
        for p in self._paths.values():
            for s in walk(p):
                print(s[0])
                print('\n'.join(['\t{}'.format(f) for f in s[2]]))

    def get(self, key, string=None, field=':', sensor_code=':', elev=':', aggr=':'):
        if string is not None:
            idx = eval('pd.IndexSlice[{}]'.format(string))
        else:
            idx = eval('pd.IndexSlice[{},{},{},{}]'.format(field, sensor_code, elev, aggr))
        return self._data[key].loc[:, idx]

    def set_meta(self, key, *args, **kwargs):
        try:
            m = self._data[key]['meta']
        except KeyError:
            m = pd.DataFrame(index=['domain','lead_day','hour'])
        self._data[key]['meta'] = pd.concat((m, self.meta(*args, **kwargs)))

    def meta(var, typ, domain, lead_day, hour):
        return pd.DataFrame([typ, domain, lead_day, hour],
                                            index=['typ', 'domain','lead_day','hour'],
                                            columns=[var])


    @staticmethod
    def compare(a, b):
        try:
            d = a.drop('data_pc', 1, 'aggr') - b.drop('data_pc', 1, 'aggr')
        except AssertionError:
            d = a - b
        if d.shape[0] > max(a.shape[0], b.shape[0]) or d.shape[1] > max(a.shape[1], b.shape[1]):
            print('Warning: DataFrames maybe not properly matched.')
        l = None
        for t, r in d[d[d!=0].count(1)!=0].iterrows():
            r.dropna(inplace=True)
            c = r[r!=0].index
            x = a.loc[t, c].to_frame()
            x.columns = pd.MultiIndex.from_product((x.columns, ['a']))
            y = b.loc[t, c].to_frame()
            y.columns = pd.MultiIndex.from_product((y.columns, ['b']))
            z = pd.concat((x, y), 1).T
            l = z if l is None else l.combine_first(z) # this makes sure there are no double columns
        return l

    def compare_all(self, file_a, file_b):
        return {k: self.compare(file_a[k], file_b[k])
                for k in set(file_a.keys()).intersection(file_b.keys())}

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            return self._paths[name]

    def __getitem__(self, value):
        idx = pd.IndexSlice
        try:
            return self.flds.loc[idx[value, :, :], :]
        except:
            return self.flds.loc[idx[:, value, :], :]

    def __del__(self):
        for d in self._data.values():
            print('closing {}'.format(d.filename))
            d.close()
