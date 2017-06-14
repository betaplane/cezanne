#!/usr/bin/env python
from os.path import join as jo
import pandas as pd
from datetime import timedelta
from configparser import ConfigParser
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
        from glob import glob
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
    def _last(d, field):
        try:
            return [
                (k, d.xs(k, 1, 'sensor_code').dropna(0, 'all').index[-1]) for k in
                d.xs(field, 1, 'field').columns.get_level_values('sensor_code').unique()
            ]
        except:
            return [None]

    def last(self, d, field):
        if isinstance(d, pd.DataFrame):
            l = self._last(d, field)
        else:
            l = [self._last(d[k], field)[0] for k in d.keys()]
        df = pd.DataFrame.from_dict(dict([x for x in l if x is not None]), orient = 'index')
        df.columns = ['last']
        df.index.name = 'sensor_code'
        return df

    def update_netcdf(self, df, var, sta=None, dt=-4, **meta):
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        h = t.replace(hour=meta['hour'])
        if h > t - timedelta(hours=dt-1):
            h = h - timedelta(days=1)
        self.OUT = WRF.OUT(paths=self.conf['wrf'].values(), from_date=h, **meta)
        n = self.OUT.netcdf([var], sta=sta)
        return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)

    def update_ceazamet(self, d, var, typ):
        from CEAZAMet import Downloader
        raw = True if typ=='ceazaraw' else False
        idx = pd.IndexSlice

        self.down = Downloader()
        self.sta, self.flds = self.down.get_stations()
        last = self.last(d, var)
        flds = self.flds.join(last, how='outer').sort_index()

        # new fields
        n = flds[flds['last'] != flds['last']].xs(var, 0, 'field', False)
        if len(n) > 0:
            b = self.down.get_field(var, n, raw=raw)
            if not raw:
                d = b.combine_first(d)

        # old fields
        f = flds.loc[idx[:, :, last.index], :]
        a = self.down.get_field(var, f, raw=raw)

        dt = timedelta(seconds=1) # because time slice is inclusive of upper boundary
        if raw:
            if isinstance(d, dict):
                d = {k: v.combine_first(d[k]) for k, v in a.items()}
            else:
                kd = dict([(i[2], i[0]) for i in flds[flds['last']==flds['last']].index.tolist()])
                d = {k: v.combine_first(d[kd[k]]) for k, v in a.items}
            if len(n) > 0:
                d.update(b.items())
        else:
            d = a.combine_first(d)
        return d

    def ceazamet_get(self, var, raw=False, update_fields=False):
        from CEAZAMet import Downloader
        self.down = Downloader()
        if update_fields:
            self.sta, self.flds = self.down.get_stations()
        return self.down.get_field(var, self.flds, raw=raw)

    @staticmethod
    def merge(hdf, d):
        for k, v in d.items():
            st = v.columns.get_level_values('station').unique()
            if len(st) != 1:
                raise Exception('DataFrame with more than one station.')
            try:
                hdf[st[0]] = v.combine_first(hdf[st[0]])
            except KeyError:
                hdf[st[0]] = v


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
        from os import walk
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

def update_meta(d):
    """Update column headings to newest format."""
    for k in d.keys():
        x = d[k]
        x.columns.names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
        d[k] = x
