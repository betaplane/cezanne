from helpers import config
from glob import glob
from datetime import datetime, timedelta
from data.interpolate import BilinearInterpolator
from importlib import import_module
from collections import namedtuple
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import os, re


def align_times(wrf, df):
    # necessary because some timestamps seem to be slightly off-round hours
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    xt = xr.DataArray(pd.Series(xt.values).dt.round('h'), coords=xt.coords).unstack('t')
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    return xr.DataArray(np.stack([np.where(idx>=0, df[c].values[idx].squeeze(), np.nan) for c in df.columns], 2),
                     coords = [wrf.coords['start'], wrf.coords['Time'], ('column', df.columns)])


# NOTE: if the 'exact' match to a wrfout timestamp is needed, use the 'time' columns on the resulting DataFrame
# - those give the original timestamps of the data points
def raw_bin(df, times, wrf_freq=pd.Timedelta(1, 'h'), label='center', tol=None):
    if tol is None:
        tol = np.diff(df.index).min() # frequency
    d0 = pd.Timedelta(0)
    dta = {
        'center': wrf_freq / 2,
        'right': wrf_freq
    }[label]
    dtb = {
        'center': dta,
        'right': d0
    }[label]
    # start-interval indexes
    a = df.index.get_indexer(times - dta, 'nearest', tolerance=tol)
    t = times[a>-1]
    da, a = df.index[a[a>-1]] - t + dta, a[a>-1]# da neg if matched index BEFORE t-dt
    # if exact match OR match BEFORE, take next index (since time label at end of interval)
    # otherwise, leave as-is
    a += (da <= d0)

    B = pd.DatetimeIndex(t + dtb)
    if B[-1] > df.index[-1]:
        a, t, B, da = a[:-1], t[:-1], B[:-1], da[:-1]
    C = B + tol

    def get(i):
        c = df.iloc[i].copy()
        idx = (c.index < C)
        c[~idx] = np.nan
        return idx, c

    idx, c = get(a)
    c['weights'] = c.index - t + dta
    c['time'] = c.index
    c.index = t
    d = [c]
    while np.any(idx):
        a += 1
        try:
            idx, c = get(a)
        except IndexError:
            assert len(d) > 2
            break
        if np.any(c.notnull()):
            c['weights'] = np.vstack((c.index, t + dtb)).min(0) - df.index[a-1]
            c['time'] = c.index
            c.index = t
            d.append(c)

    d = pd.concat(d, 1, keys=range(len(d)))
    return d

def raw_stats(df):
    w = df.xs('weights', 1, 'station')

    # some checks ====================
    u, c = np.unique(w, return_counts=True)
    # most frequent sampling interval (i.e. weight for averages)
    W = u[c.argmax()]
    print('\nirregularities: {}\n'.format(dict(zip(u[u!=W], c[u!=W]))))

    # shorter intervals should only occur on edges
    i, j = np.where(w<W)
    assert set(j) - {0, w.shape[1]-1} == set()
    # longer intervals only inside bin
    i, j = np.where(w>W)
    j = set(j)
    if j != set():
        assert min(j) > 0
        assert max(j) < w.shape[1] - 1
    u, c = np.unique(i, return_counts=True)
    k = df.index[u[c>1]] # rows with more than one missing values inside the bin
    if len(k) > 0:
        df = df.drop(k)
        w = df.xs('weights', 1, 'station')
        print('{} rows dropped due to missing values.'.format(len(k)))
    # ================================

    ave = df.xs('ave', 1, 'aggr').values
    # reset all 'interior' weights to the modal value
    w.loc[df.index, np.arange(1, w.shape[1])] = W
    w = w.values.astype('timedelta64[m]').astype(float)
    ave = (ave * w).sum(1) / (w * np.isfinite(ave).astype(int)).sum(1)

    x = np.vstack((
        ave,
        df.xs('min', 1, 'aggr').values.min(1),
        df.xs('max', 1, 'aggr').values.max(1)
    ))
    return pd.DataFrame(x.T, index=df.index, columns=['ave', 'min', 'max'])


class Hbin:
    """Hourly binning of raw station data."""
    @staticmethod
    def weights(minutes, split_min):
        m = sorted(np.r_[minutes, split_min])
        d = np.diff(np.r_[m[-1], m]) % 60
        i = m.index(split_min)
        return dict(zip(m, d)), {m[(i+1) % len(m)]: d[i]}

    @staticmethod
    def ave(df, weights):
        w = [weights.get(m, 0) for m in df.index.minute]
        x = df.values.flatten()
        d, n = (np.vstack((x, np.isfinite(x))) * w).sum(1)
        return pd.Series([d, n], index=['sum', 'weight'])

    @classmethod
    def bin(cls, df, label='center', start=None):
        kwargs = {'rule': '60T', 'closed': 'right', 'label': 'right'}
        base = {'center': 30, 'right': 0}[label]
        kwargs.update(base=base)
        loffs = pd.offsets.Minute(-base)
        ave = df.xs('ave', 1, 'aggr')

        m = np.unique(df.index.minute)
        if len(m)==1:
            if (label=='right' and m[0]==0) or (label=='center' and m[0]==30):
                df = df.copy()
                df.columns = df.columns.get_level_values('aggr')
                return df
        w, e = cls.weights(m, {'center': 30, 'right': 0}[label])
        a = ave.resample(loffset=loffs, **kwargs).apply(cls.ave, weights=w)
        b = ave.resample(loffset=loffs - pd.offsets.Hour(1), **kwargs).apply(cls.ave, weights=e)
        d = a.add(b, fill_value=0)
        ave = d['sum'] / d['weight']
        # apparently, reusing a resampler leads to unpredictble results
        mi = df.xs('min', 1, 'aggr').resample(loffset=loffs, **kwargs).min().iloc[:, 0]
        ma = df.xs('max', 1, 'aggr').resample(loffset=loffs, **kwargs).max().iloc[:, 0]
        x = pd.concat((ave, mi, ma), 1, keys=['ave', 'min', 'max'])
        return x if start is None else x.loc[start:]

class WRFR(config.WRFop):
    utc_delta = pd.Timedelta(-4, 'h')
    update_overlap = pd.Timedelta(1, 'd')
    time_dim_coord = 'XTIME'
    wrf_dim_order = ('start', 'station', 'Time')
    K = 273.15

    def __init__(self, data=None, loaded=None, dir_pat=None, copy=None):
        assert (data is not None) or (copy is not None) or (loaded is not None), "One of 'data', 'loaded' or 'copy' needs to be specified"
        if copy is not None:
            self.__dict__ = copy.__dict__
            return

        dirpat = re.compile(self.directory_re if dir_pat is None else dir_pat)
        self.wrfout_re, self.wrfxtrm_re = re.compile(self.wrfout_re), re.compile(self.wrfxtrm_re)

        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            sta = S['stations']
            self.flds = S['fields']

        self.loaded = loaded
        if data is not None:
            data = {k: v for k, v in data.items() if (v is not None and v.shape[0]>0)}
            print('data with length > 0: {}'.format(len(data)))
            mi, ma = zip(*[(lambda x: (x.min(), x.max()))(df.dropna().index) for df in data.values()])
            self.start, end = min(mi), max(ma)
            self.raw = data
        else:
            # when updating, we use two different starting points for station data and WRF files
            # (b/c WRF files are not subject to changes once written out, unlike the stations db)
            # NOTE: my binning will cause typically 2 hourly timesteps at the end/beginning to be sensitive to new data
            self.start_bin = loaded.binned.dropna(0, 'all').index.max() - self.update_overlap
            raw = {k: v.loc[self.start_bin:] for k, v in loaded.raw.items() if v is not None}
            self.raw = {k: v for k, v in raw.items() if v.shape[0]>0}
            self.start = pd.Timestamp(loaded.intp['start'].max().item()) + pd.Timedelta(1, 'd')
            end = max([df.dropna().index.max() for df in self.raw.values()])
            print('start-wrf: {}; end-wrf: {}; start-binning: {}'.format(self.start, end, self.start_bin))

        s = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        self.sta = sta.loc[[s[k] for k in self.raw.keys()]]

        dirs = []
        for p in self.paths:
            try:
                dirs.extend([os.path.join(p, d) for d in os.listdir(p) if dirpat.search(d)])
            except: pass
        dirs = sorted(dirs, key=lambda s:s[-10:])

        def test(d):
            t = datetime.strptime(d[-10:],'%Y%m%d%H')
            try:
                assert os.path.isdir(d)
                assert t >= self.start
                assert t <= end
            except: return False
            else: return True

        self.dirs = [d for d in dirs if test(d)]

        if data is not None:
            # insert additional directories in front whose simulations overlap with data
            for d in dirs[dirs.index(self.dirs[0])-1::-1]:
                try:
                    out = [os.path.join(d, f) for f in os.listdir(d) if self.wrfout_re.search(f)]
                    ds = xr.open_dataset(sorted(out)[-1])
                    if ds[self.time_dim_coord].values.max() >= start.asm8:
                        self.dirs.insert(0, d)
                    else:
                        break
                except: pass
                finally: ds.close()

        # there's a list with simulations with errors in config_mod
        for s in self.skip:
            try:
                self.dirs.remove(s)
            except: pass


    def wrf_files(self, wrfout_vars, wrfxtrm_vars=None):
        from functools import reduce
        # always interpolate to all historically active stations - since, why not
        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            sta = reduce(lambda i, j: S[i].combine_first(S[j]),
                         sorted([k for k in S.keys() if re.search('station', k)], reverse=True))

        out = [f for f in os.listdir(self.dirs[0]) if self.wrfout_re.search(f)]
        with xr.open_dataset(os.path.join(self.dirs[0], out[0])) as ds:
            itp = BilinearInterpolator(ds, stations=sta)

        o, x = [], []
        for d in tqdm(self.dirs):
            dl = os.listdir(d)
            out = [os.path.join(d, f) for f in dl if self.wrfout_re.search(f)]
            if len(out) == 0:
                continue
            elif len(out) == 1:
                with xr.open_dataset(out[0]) as ds:
                    oi = ds[wrfout_vars].apply(itp.xarray)
            else:
                with xr.open_mfdataset(out) as ds:
                    oi = ds[wrfout_vars].apply(itp.xarray)

            t = oi.coords[self.time_dim_coord]
            oi.coords['start'] = t.min()
            t += self.utc_delta
            oi.coords[self.time_dim_coord] = t
            o.append(oi)

            if wrfxtrm_vars is not None:
                xtrm = [os.path.join(d, f) for f in dl if self.wrfxtrm_re.search(f)]
                if len(xtrm) > 1:
                    with xr.open_mfdataset(xtrm) as ds:
                        xi = ds[wrfxtrm_vars].apply(itp.xarray)
                else:
                    with xr.open_dataset(xtrm[0]) as ds:
                        xi = ds[wrfxtrm_vars].apply(itp.xarray)

                xi.coords[self.time_dim_coord] = t
                xi.coords['start'] = oi.coords['start']
                x.append(xi)

        self.intp = xr.concat(o, 'start').transpose(*self.wrf_dim_order)
        if wrfxtrm_vars is not None:
            self.intp = xr.merge((self.intp, xr.concat(x, 'start').transpose(*self.wrf_dim_order)))

    @staticmethod
    def combine_first(a, b):
        c = a.combine_first(b)
        c.coords['XTIME'] = a['XTIME'].combine_first(b['XTIME'])
        return c

    def stats(self, wrfout, wrfxtrm, ceazamet):
        self.wrf_files([wrfout], list(wrfxtrm.keys()))

        try:
            # if updating
            st = self.loaded.binned.dropna(0, 'all').index.max() - self.update_overlap / 2
        except:
            st = None
        sta = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        X, B = {}, {}
        for s, df in tqdm(self.raw.items()):
            d_mid = Hbin.bin(df, label='center', start=st).dropna(0, 'all')
            d_end = Hbin.bin(df, label='right', start=st).dropna(0, 'all')
            x = self.intp.sel(station=sta[s])
            try: # when updating, we might need some earlier wrf-interpolations
                t = min(d_mid.index.min(), d_end.index.min()).asm8
                i = min(np.where(self.loaded.intp['XTIME'] > t)[x['XTIME'].dims.index('start')])
                x = self.combine_first(x, self.loaded.intp.sel(station=sta[s]).isel(start=slice(i, None)))
            except: pass
            dm = align_times(x, d_mid).sel(column='ave') + self.K
            dm['column'] = 'point'
            de = align_times(x, d_end) + self.K
            X[sta[s]] = xr.concat((
                xr.concat([x[k] - de.sel(column=v) for k, v in wrfxtrm.items()], 'column'),
                x[wrfout] - dm
            ), 'column')
            B[s] = d_end
        self.binned = self.dict2frame(B)

        # NOTE: In the wrfxtrm files, the first data point of a simulation run is always 0!
        # nan it out here
        x = xr.concat(X.values(), 'station')
        d = {
            'station': slice(None),
            'column': x.indexes['column'].get_indexer(['ave', 'max', 'min']),
            'start': slice(None),
            'Time': 0,
        }
        x[tuple([d[c] for c in x.dims])] = np.nan
        self.err = x.to_dataset('column').transpose(*self.wrf_dim_order)

    def dict2frame(self, B):
        names = ['station', 'field', 'sensor_code', 'elev', 'aggr']
        d = self.flds['sensor_code'].reset_index().join(self.sta, on='station')
        def df(k, v):
            df = v.copy()
            s = d[d['sensor_code']==k][names[:-1]].values[0]
            df.columns = pd.MultiIndex.from_tuples([np.r_[s, [c]] for c in v.columns])
            df.columns.names = names
            return df
        return pd.concat([df(k, v) for k, v in B.items()], 1)

    @staticmethod
    def nc_overwrite(data, filename):
        # There's some weird issue with overwriting a previously-opened file in xarray / netCDF4
        # (no matter how much I try to close the dataset or use a 'with' context).
        # I can't reproduce it reliably, except *this* piece of code without the renaming shenanigans
        # *always* fails. Hence the renaming shenanigans.
        fn = '{}_overwrite.nc'.format(os.path.splitext(filename)[0])
        data.to_netcdf(fn, mode='w')
        os.rename(fn, filename)

    def store(self):
        if self.loaded is None:
            print('no pre-loaded data')
            intp = self.intp
            err = self.err
            B = self.binned
        else:
            st = self.start_bin + self.update_overlap / 2
            B = self.binned.loc[st:].combine_first(self.loaded.binned)
            # need to treat XTIME separately because combine_first seems to drop unused coordinates
            intp = self.combine_first(self.intp, self.loaded.intp)
            err = self.combine_first(self.err, self.loaded.err)

        var_code = self.binned.columns.get_level_values('field').unique().item()
        B.to_hdf(config.CEAZAMet.raw_data, 'raw/binned/end/{}'.format(var_code))
        self.nc_overwrite(intp, self.wrf_intp)
        self.nc_overwrite(err, self.wrf_err)

    @classmethod
    def load(cls, var_code, limit=False):
        nt = namedtuple('data', ['binned', 'raw', 'intp', 'err'])
        l = int(limit)
        ds = xr.open_dataset(cls.wrf_intp)
        intp = ds.isel(start=slice(*[(None,), (-10, None)][l])).load()
        ds.close()
        try:
            ds = xr.open_dataset(cls.wrf_err)
            err = ds.isel(start=slice(*[(None,), (-10, None)][l])).load()
            ds.close()
        except:
            err = None
        with pd.HDFStore(config.CEAZAMet.raw_data) as S:
            node = S.get_node('raw/{}'.format(var_code))
            raw = {k: S[v._v_pathname].iloc[slice(*[(None,), (-1000, None)][l])]
                   for k, v in node._v_children.items()}
            binned = S['raw/binned/end/{}'.format(var_code)].iloc[slice(*[(None,), (-100, None)][l])]
        return nt(binned, raw, intp, err)

    def map_plot(self, coq):
        cpl = import_module('plots')
        plt = import_module('matplotlib.pyplot')
        gs = import_module('matplotlib.gridspec')
        # coq = cpl.Coquimbo()
        fig = plt.figure(figsize=(15, 4))
        g = gs.GridSpec(1, 32, left=.05, right=.95)

        x = self.wrf.isel(Time=slice(24, 48))
        dims = ('Time', 'start')
        def f(col, func):
            return pd.DataFrame(getattr(x[col], func)(dims).values, index=x.station, columns=[col])
        df = pd.concat((f('ave', 'mean'), f('point', 'mean'), f('min', 'min'), f('max', 'max')), 1)
        lonlat = self.sta[['lon', 'lat']].values.T

        plt.set_cmap('seismic')
        coq.plotrow(df[['ave', 'point']], g[0, 1:11], cbar='left', ylabels=False, cbar_kw={'center': True, 'cax': g[0, 0]})
        plt.set_cmap('Blues_r')
        coq.plotrow(df[['min']], g[0, 13:18], cbar='left', ylabels=False, cbar_kw={'cax': g[0, 12]})
        plt.set_cmap('Reds')
        coq.plotrow(df[['max']], g[0, 20:25], cbar='left', ylabels=False, cbar_kw={'cax': g[0, 19]})
        plt.set_cmap('plasma')
        c = f('ave', 'count')
        c.columns = ['count']
        coq.plotrow(c, g[0, 26:31], cbar_kw={'cax': g[0, 31]})

    @classmethod
    def plot_station(cls, data, sensor_code, time, prev=None):
        plt = import_module('matplotlib.pyplot')
        gs = import_module('matplotlib.gridspec')
        flds = pd.read_hdf(config.CEAZAMet.meta_data, 'fields')
        sta = {k: v for (v, _), k in flds['sensor_code'].items()}
        colrs = plt.get_cmap('Set2').colors
        fig = plt.figure()
        g = gs.GridSpec(3, 1)

        d = data.binned.xs(sensor_code, 1, 'sensor_code').dropna().resample('h').asfreq()
        try:
            df = cls.dataframe(data.wrf, time)[sta[sensor_code]].resample('h').asfreq()
        except: pass

        ax = fig.add_subplot(g[:2, 0])
        ax.set_title(sta[sensor_code])
        bx = fig.add_subplot(g[2, 0], sharex=ax)
        for i, (a, b) in enumerate([('ave', 'T2MEAN'), ('max', 'T2MAX'), ('min', 'T2MIN')]):
            ax.plot(d.xs(a, 1, 'aggr'), color=colrs[i], label='obs {}'.format(a))
            try:
                ax.plot(df[b] - cls.K, color=colrs[i+3], label='WRF {}'.format(a))
                bx.plot(df[a], color=colrs[i+3], label='err {}'.format(a))
            except: pass

        start = min([df.dropna().index.min() for df in data.raw.values()])
        if prev is not None:
            D, Df = prev.binned, cls.dataframe(prev.wrf, time)
            dt = pd.Timedelta(2, 'd')
            x = D.xs(sensor_code, 1, 'sensor_code').xs('ave', 1, 'aggr')
            ax.plot(x.loc[start-dt:], color=colrs[6], label='obs old')

        ax.axvline(start, color='r')
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.grid(axis='x')
        ax.legend()

        bx.axvline(start, color='r')
        bx.grid(axis='x')
        bx.legend()

    @staticmethod
    def dataframe(nc, time):
        def nat(df):
            return df[df['XTIME']==df['XTIME']]
        wrf = nat(nc.isel(Time=time).to_dataframe())
        wrf = wrf.reset_index().pivot('XTIME', 'station', list(nc.data_vars))
        return wrf.sort_index(1).swaplevel(axis=1)

    def check_overlap(self, other, aggr='ave'):
        a = other.xs(aggr, 1, 'aggr')
        b = self.binned.xs(aggr, 1, 'aggr')
        c = a.columns.get_level_values(0).intersection(b.columns.get_level_values(0))
        return pd.concat([(a[s] - b[s]).dropna() for s in c], 1, keys=c).T

def update(var_code):
    from data import CEAZA
    # the CEAZA module takes care of updating *its* data (hopefully)
    f = CEAZA.Field()
    try:
        f.update(var_code, raw=True)
    except CEAZA.FieldsUpToDate:
        print('CEAZAMet data are up to date')

    D = WRFR.load(var_code)
    w = WRFR(loaded=D)
    for kwargs in config.WRFop.variables:
        w.stats(**kwargs)
    return w
    # w.store()

def compare_xr(a, b):
    eq = (a == b).sum(('Time', 'start')).to_dataframe()
    neq = (a != b) * np.isfinite(a-b)
    t = {k: neq.XTIME.values[neq[k]].min() for k in neq.data_vars}
    neq = neq.sum(('Time', 'start')).to_dataframe()
    return pd.concat((neq, eq), 1, keys=['neq', 'eq']).swaplevel(axis=1).sort_index(1), t

def compare_df(a, b):
    print('only in a: {}'.format(set(a.keys()) - b.keys()))
    print('only in b: {}'.format(set(b.keys()) - a.keys()))
    d = {}
    for k in set(a.keys()).intersection(b.keys()):
        idx = a[k].index.intersection(b[k].index)
        col = a[k].columns.intersection(b[k].columns)
        ij = [np.unique(i) for i in np.where(a[k].loc[idx, col]!=b[k].loc[idx, col])]
        ij = [i for i in ij if len(i)>0]
        d[k] = {'diff': ij if len(ij)>0 else None}

        ij = {k: v for k, v in {
            'a': a[k].index.difference(b[k].index),
            'b': b[k].index.difference(a[k].index)
        }.items() if len(v) > 0}
        d[k].update(idx=ij if len(ij) > 0 else None)
    ab = np.unique(np.hstack([list(v['idx'].keys()) for k, v in d.items() if v['idx'] is not None]))
    try:
        c = ab.item()
        idx = np.unique(np.hstack([v['idx'][c].values for k, v in d.items() if v['idx'] is not None]))
        return d, idx
    except: pass
    return d

def ex(x, var, station, day):
    x = x[var].sel(station=station, Time=slice(day*24, (day+1)*24)).stack(t=('Time', 'start')).sortby('XTIME')
    return x.XTIME, x
