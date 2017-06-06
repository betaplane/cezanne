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


conf = ConfigParser()
conf.read('data.cfg')

class Data(object):
    def __init__(self, config='data.cfg'):
        self.conf = ConfigParser()
        self.conf.read(confi)

    def open(name):
        ext = os.path.splitext(name)[1]
        p = os.path.join(conf['data']['base'], conf['data'][ext], name)
        return pd.HDFStore(p)


def netcdf(var, domain, lead_day, from_date, sta=None):
    dirs = sorted(glob(os.path.join(conf['wrf']['op1'], 'c01*')))
    dirs.extend(sorted(glob(os.path.join(conf['wrf']['op2'], 'c01*'))))
    s = (from_date + timedelta(days=lead_day)).strftime('%Y%m%d%H')
    def name(d):
        dt = datetime.strptime(os.path.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(os.path.join(d, '*_{}_*'.format(domain)))
        s = (dt + timedelta(days=lead_day)).strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(d, 'wrfout_{}_{}'.format(domain, s))
    files = [name(d) for d in dirs if d[-10:] >= s]
    files = [f for f in files if os.path.isfile(f)]
    ds = xr.open_dataset(files[0])
    if sta is not None:
        Map = Basemap(projection='lcc', **matts(ds))
        df = xr_interp(ds, var, sta, map=Map)
        with cc.ThreadPoolExecutor(max_workers=4) as exe:
            fl = {exe.submit(lambda f:xr_interp(xr.open_dataset(f), var, sta, map=Map)): f for f in files[1:]}
            for f in cc.as_completed(fl):
                df = pd.concat((df, f.result()), 0)
                print(fl[f])
    else:
        df = xr.open_dataset(files[0])[var]
        with cc.ThreadPoolExecutor(max_workers=4) as exe:
            fl = {exe.submit(lambda f: xr.open_dataset(f)[var]): f for f in files[1:]}
            for f in cc.as_completed(fl):
                df = xr.concat((df, f.result()), 'Time')
                print(fl[f])
    return df


def append(df, *args, hour=12, dt=-4, **kwargs):
    t = df.dropna(0, 'all').index[-1].to_datetime()
    h = t.replace(hour=hour)
    if h > t - timedelta(hours=dt-1):
        h = h - timedelta(days=1)
    n = netcdf(*args, **kwargs, from_date=h)
    return pd.concat((df[:n.index[0] - timedelta(minutes=1)], n), 0)
