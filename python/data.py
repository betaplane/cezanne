#!/usr/bin/env python
from os.path import join as jo
from os import walk
import pandas as pd
from datetime import timedelta
from glob import glob
from configparser import ConfigParser
from CEAZAMet import Fetcher, NoNewStationError
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
        try:
            self.sta = self._sta['stations']
            self.flds = self._sta['fields']
        except:
            pass

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

    def update_netcdf(self, df, var, sta=None, dt=-4, **meta):
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        h = t.replace(hour=meta['hour'])
        if h > t - timedelta(hours=dt-1):
            h = h - timedelta(days=1)
        self.OUT = WRF.OUT(paths=self.conf['wrf'].values(), from_date=h, **meta)
        n = self.OUT.netcdf([var], sta=sta)
        return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)

    def update_ceazamet(self, df, var, typ):
        raw = True if typ=='ceazaraw' else False
        t = df.dropna(0, 'all').index[-1].to_pydatetime() - timedelta(days=2)
        self.fetch = Fetcher()
        try:
            sta, flds = self.fetch.get_stations(self.sta)
        except NoNewStationError:
            pass
        else:
            n = set(flds.index.get_level_values('sensor_code')) - set(df.columns.get_level_values('sensor_code'))
            f = flds.loc[pd.IndexSlice[list(n), var, :], :]
            b = self.fetch.get_field(var, f, raw=raw)
            d = pd.concat((d, b), 1)
            self._sta['stations'] = sta
            self._sta['fields'] = flds
            self.sta = sta
            self.flds = flds
        a = self.fetch.get_field(var, self.flds, from_date=t.date(), raw=raw)
        d = pd.concat((df[:a.index[0] - timedelta(seconds=1)], a), 0)
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
