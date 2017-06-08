#!/usr/bin/env python
from glob import glob
import xarray as xr
from functools import partial
from interpolation import xr_interp
import concurrent.futures as cc
from timeit import default_timer as timer
from datetime import datetime, timedelta


class Fetcher(object):
    def __init__(self, meta, conf, from_date=None):
        name = partial(self.name, meta['domain'],  meta['lead_day'])
        dirs = sorted(glob(os.path.join(conf['wrf']['op1'], 'c01*')))
        dirs.extend(sorted(glob(os.path.join(conf['wrf']['op2'], 'c01*'))))
        if from_date is None:
            files = [name(d) for d in dirs if d[-2:] == '{:02}'.format(meta['hour'])]
        else:
            dh = (from_date + timedelta(days=meta['lead_day'])).strftime('%Y%m%d%H')
            files = [name(d) for d in dirs if d[-10:] >= dh]
        files = [f for f in files if os.path.isfile(f)]

    @staticmethod
    def name(dom, lead, d):
        dt = datetime.strptime(os.path.split(d)[1], 'c01_%Y%m%d%H')
        f = glob(os.path.join(d, '*_{}_*'.format(dom)))
        s = (dt + timedelta(days=lead)).strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(d, 'wrfout_{}_{}'.format(dom, s))

    def netcdf(self, var, sta=None, dt=-4):
        start = timer()
        if sta is not None:
            df, Map = xr_interp(files[0], var, sta)
            with cc.ThreadPoolExecutor(max_workers=16) as exe:
                fl = {exe.submit(lambda x:xr_interp(xr.open_dataset(x), var, sta, map=Map, dt=dt), f):
                      f for f in files[1:]}
                for f in cc.as_completed(fl):
                    df = pd.concat((df, f.result()), 0)
                    self.data = df
                    print(fl[f])
            df = sort_index(inplace=True)
        else:
            df = None
            with cc.ThreadPoolExecutor(max_workers=16) as exe:
                fl = {exe.submit(xr.open_dataset, f): f for f in files}
                for f in cc.as_completed(fl):
                    ds = f.result()
                    df = xr.concat((df, ds[var]), 'Time')
                    self.data = df
                    ds.close()
                    print(fl[f])
            df = df[df['XTIME'].argsort(), :, :]
            df['XTIME'] = df['XTIME'] + np.timedelta64(dt, 'h')
        print(timer() - start)
        return df

