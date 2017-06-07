#!/usr/bin/env python
import os
import pandas as pd
from datetime import timedelta
from configparser import ConfigParser
from CEAZAMet as import Fetcher, NoNewStationError
from WRF import Fetcher as WRF


class Data(object):
    def __init__(self, config='data.cfg'):
        self.conf = ConfigParser()
        self.conf.read(config)
        self._data = {}
        self.meta = {}
        self.open('_sta', self.conf['data']['sta'])
        self.sta = self._sta['stations']
        self.flds = self._sta['fields']

    def open(self, key, name):
        ext = os.path.splitext(name)[1]
        p = os.path.join(self.conf['data']['base'], self.conf['data'][ext], name)
        self._data[key] = pd.HDFStore(p)

    def append(self, key, var, sta=None, dt=-4):
        d = self._data[key]
        df = d[var]
        m = d['meta'][var].to_dict()
        typ = m.pop('type')
        if typ == 'netcdf':
            D = self._append_netcdf(df, var, m, sta, dt)
        elif typ == 'ceazamet' or type == 'ceazaraw':
            D = self._append_ceazamet(df, var, typ)
        self._data[key][var] = D

    def _append_netcdf(self, df, var,  meta, sta=None, dt=-4):
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        h = t.replace(hour=m['hour'])
        if h > t - timedelta(hours=dt-1):
            h = h - timedelta(days=1)
        self.fetch = WRF(meta, self.conf, h)
        n = self.fetch.netcdf(var, sta=sta)
        return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)

    def _append_ceazamet(self, df, var, typ):
        raw = True if typ=='ceazaraw' else False
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
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


    def set_meta(self, key, domain, lead_day, hour):
        try:
            m = self._data[key]['meta']
        except KeyError:
            m = pd.DataFrame(index=['domain','lead_day','hour'])
        self._data[key]['meta'] = pd.concat((m, pd.DataFrame([domain, lead_day, hour],
                                            index=['domain','lead_day','hour'],
                                            columns=['key'])), 1)
    @staticmethod
    def compare(a, b):
        d = a.drop('data_pc', 1, 'aggr') - b.drop('data_pc', 1, 'aggr')
        l = None
        for t, r in d[d[d!=0].count(1)!=0].iterrows():
            r.dropna(inplace=True)
            c = r[r!=0].index
            x = a.loc[t, c].to_frame()
            x.columns = pd.MultiIndex.from_product((x.columns, ['a']))
            y = b.loc[t, c].to_frame()
            y.columns = pd.MultiIndex.from_product((y.columns, ['b']))
            z = pd.concat((x, y), 1)
            l = pd.concat((l, z), 0)
        return l

    def __getattr__(self, name):
        return self._data[name]

    def __del__(self):
        for d in self._data.values():
            d.close()
