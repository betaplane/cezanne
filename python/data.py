#!/usr/bin/env python
from glob import glob
import os
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from configparser import ConfigParser
from mpl_toolkits.basemap import Basemap
from mapping import matts
from interpolation import xr_interp
import concurrent.futures as cc
from timeit import default_timer as timer
import CEAZAMet as cm


conf = ConfigParser()
conf.read('data.cfg')


class Data(object):
    def __init__(self, config='data.cfg'):
        self.conf = ConfigParser()
        self.conf.read(config)
        self._sta = self.conf['data']['sta']
        self.sta = self._sta['sta']
        self._data = {}
        self.meta = {}

    def open(self, key, name):
        ext = os.path.splitext(name)[1]
        p = os.path.join(conf['data']['base'], conf['data'][ext], name)
        self._data[key] = pd.HDFStore(p)

    def append(self, key, var, sta=None, dt=-4):
        d = self._data[key]
        df = d[var]
        m = d['meta'][var].to_dict()
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        typ = m.pop('type')
        if typ == 'netcdf':
            return self._append_netcdf(df, var, m, sta, dt)
        elif type == 'ceazamet' or type == 'ceazaraw':
            return self._append_ceazamet(df, var, typ)

    def _append_netcdf(self, df, var,  meta, sta=None, dt=-4):
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        h = t.replace(hour=m['hour'])
        if h > t - timedelta(hours=dt-1):
            h = h - timedelta(days=1)
        n = netcdf(var, from_date=h, sta=sta, **meta)
        return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)

    def _append_ceazamet(self, df, var, typ):
        t = df.dropna(0, 'all').index[-1].to_pydatetime()
        sta = cm.get_stations(self.sta)
        n = cm.get_field(var, self.sta, from_date=t.date(), raw=True if typ=='ceazaraw' else False)
        n = pd.concat((df[:n.index[0] - timedelta(seconds=1)], n), 0)
        if sta:
            m = cm.get_field(var, sta, raw=True if typ=='ceazaraw' else False)
            n = pd.concat((n, m), 1)


    def meta(self, key, domain, lead_day, hour):
        try:
            m = self._data[key]['meta']
        except KeyError:
            m = pd.DataFrame(index=['domain','lead_day','hour'])
        self._data[key]['meta'] = pd.concat((m, pd.DataFrame([domain, lead_day, hour],
                                            index=['domain','lead_day','hour'],
                                            columns=['key'])), 1)


    def __getattr__(self, name):
        return self._data[name]

    def __del__(self):
        for d in self._data.values():
            d.close()



def netcdf(var, domain, lead_day, from_date=None, hour=None, sta=None, dt=-4):
    start = timer()
    def name(d):
        dt = datetime.strptime(os.path.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(os.path.join(d, '*_{}_*'.format(domain)))
        s = (dt + timedelta(days=lead_day)).strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(d, 'wrfout_{}_{}'.format(domain, s))

    dirs = sorted(glob(os.path.join(conf['wrf']['op1'], 'c01*')))
    dirs.extend(sorted(glob(os.path.join(conf['wrf']['op2'], 'c01*'))))
    if from_date is None:
        files = [name(d) for d in dirs if d[-2:] == '{:02}'.format(hour)]
    else:
        dh = (from_date + timedelta(days=lead_day)).strftime('%Y%m%d%H')
        files = [name(d) for d in dirs if d[-10:] >= dh]
    files = [f for f in files if os.path.isfile(f)]
    ds = xr.open_dataset(files[0])
    if sta is not None:
        Map = Basemap(projection='lcc', **matts(ds))
        df = xr_interp(ds, var, sta, map=Map)
        with cc.ThreadPoolExecutor(max_workers=16) as exe:
            fl = {exe.submit(lambda x:xr_interp(xr.open_dataset(x), var, sta, map=Map, dt=dt), f):
                  f for f in files[1:]}
            for f in cc.as_completed(fl):
                df = pd.concat((df, f.result()), 0)
                print(fl[f])
        df.sort_index(inplace=True)
    else:
        df = xr.open_dataset(files[0])[var]
        with cc.ThreadPoolExecutor(max_workers=16) as exe:
            fl = {exe.submit(xr.open_dataset, f): f for f in files[1:]}
            for f in cc.as_completed(fl):
                ds = f.result()
                df = xr.concat((df, ds[var]), 'Time')
                df.close()
                print(fl[f])
        df = df[df['XTIME'].argsort(), :, :]
        df['XTIME'] = df['XTIME'] + np.timedelta64(dt, 'h')
    print(timer() - start)
    return df


