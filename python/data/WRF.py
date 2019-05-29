from helpers import config, sta
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
    """Align :class:`~pandas.DataFrame` with station data (as produced by :class:`.CEAZA.CEAZAMet`) with a concatenated (and interpolated to station locations) netCDF file as produced by this packages concatenators (:class:`threadWuRF.CC`, :class:`mpiWuRF.CC`). For now, works with a single field.

    :param wrf: The DataArray or Dataset containing the concatenated and interpolated (to station locations) WRF simulations (only dimensions ``start`` and ``Time`` and coordinate ``XTIME`` are used).
    :type wrf: :class:`~xarray.DataArray` or :class:`~xarray.Dataset`
    :param df: The DataFrame containing the station data (of the shape returned by :meth:`.CEAZA.Downloader.get_field`).
    :type df: :class:`~pandas.DataFrame`
    :returns: DataArray with ``start`` and ``Time`` dimensions aligned with **wrf**.
    :rtype: :class:`~xarray.DataArray`

    """
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    # necessary because some timestamps seem to be slightly off-round hours
    xt = xr.DataArray(pd.Series(xt.values).dt.round('h'), coords=xt.coords).unstack('t')
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    try:
        cols = df.columns.get_level_values('station').intersection(wrf.station)
        cols_name = 'station'
    except:
        cols = df.columns
        cols_name = 'column'
    return xr.DataArray(np.stack([np.where(idx>=0, df[c].values[idx].squeeze(), np.nan) for c in cols], 2),
                     coords = [wrf.coords['start'], wrf.coords['Time'], (cols_name, cols)])


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
    def bin(cls, df, label='center'):
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
        return pd.concat((ave, mi, ma), 1, keys=['ave', 'min', 'max'])

class WRFD(config.WRFop):
    utc_delta = pd.Timedelta(-4, 'h')
    time_dim_coord = 'XTIME'
    K = 273.15

    def __init__(self, dir_pat='c01'):
        dirpat = re.compile(dir_pat)
        dirs = []
        for p in self.paths:
            try:
                dirs.extend([os.path.join(p, d) for d in os.listdir(p)])
            except: pass

        self.dirs = sorted([d for d in dirs if os.path.isdir(d) and dirpat.search(d)], key=lambda s:s[-10:])
        # there's a list with simulations with errors in config_mod
        for s in self.skip:
            self.dirs.remove(s)

    def files(self, file_pat='wrfout_d03', lead=None):
        self.filepat = re.compile(file_pat)
        files = []
        for d in self.dirs:
            flist = [os.path.join(d, f) for f in os.listdir(d) if self.filepat.search(f)]
            if len(flist) == 1 or lead is None:
                files.extend(flist)
            else:
                t = datetime.strptime(d[-10:],'%Y%m%d%H')
                s = (t + timedelta(days=lead)).strftime('%Y-%m-%d')
                files.extend([f for f in flist if re.search(s, f)])

        return sorted(files, key=os.path.basename)

    # NOTE: not corrected for GMT diffs yet
    def normal(self, var_list, stations=sta, **kwargs):
        files = self.files(**kwargs)
        with xr.open_dataset(files[0]) as ds:
            itp = BilinearInterpolator(ds, stations=stations)
        x = []
        for f in tqdm(files):
            with xr.open_dataset(f) as ds:
                x.append(ds[var_list].apply(itp.xarray))
        return xr.concat(x, 'start')

    def dask(self, var_list, stations=sta, **kwargs):
        ds = xr.open_mfdataset(self.files(**kwargs))
        itp = BilinearInterpolator(ds, stations=stations)
        x = ds[var_list].apply(itp.xarray)

class WRFR(config.WRFop):
    utc_delta = pd.Timedelta(-4, 'h')
    time_dim_coord = 'XTIME'
    K = 273.15

    def __init__(self, data=None, dir_pat='c01', loaded=None, copy=None):
        if copy is not None:
            self.__dict__ = copy.__dict__
            return
        dirpat = re.compile(dir_pat)
        self.wrfout_re, self.wrfxtrm_re = re.compile(self.wrfout_re), re.compile(self.wrfxtrm_re)
        data = {k: v for k, v in data.items() if (v is not None and v.shape[0]>0)}
        print('data with length > 0: {}'.format(len(data)))

        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            sta = S['stations']
            self.flds = S['fields']

        mi, ma = zip(*[(lambda x: (x.min(), x.max()))(df.dropna().index) for df in data.values()])
        self.start, end = min(mi), max(ma)
        s = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        self.sta = sta.loc[[s[k] for k in data.keys()]]
        self.raw = data
        self.loaded = loaded

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

        self.dirs = sorted([d for d in dirs if test(d)], key=lambda s:s[-10:])

        # insert additional directories in front whose simulations overlap with data
        for d in dirs[dirs.index(self.dirs[0])-1::-1]:
            try:
                out = [os.path.join(d, f) for f in os.listdir(d) if self.wrfout_re.search(f)]
                ds = xr.open_dataset(sorted(out)[-1])
                if ds[self.time_dim_coord].values.max() >= self.start.asm8:
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
        out = [f for f in os.listdir(self.dirs[0]) if self.wrfout_re.search(f)]
        with xr.open_dataset(os.path.join(self.dirs[0], out[0])) as ds:
            itp = BilinearInterpolator(ds, stations=self.sta)

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

        wrf = xr.concat(o, 'start')
        self.wrf = xr.merge((wrf, xr.concat(x, 'start'))) if wrfxtrm_vars is not None else wrf

    def stats(self, wrfout, wrfxtrm, ceazamet):
        self.wrf_files([wrfout], list(wrfxtrm.keys()))

        sta = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        t = np.unique(self.wrf[self.time_dim_coord])
        X, B = {}, {}
        for s, df in tqdm(self.raw.items()):
            x = self.wrf.sel(station=sta[s])
            d_mid = Hbin.bin(df, label='center')
            dm = align_times(x, d_mid).sel(column='ave') + self.K
            dm['column'] = 'point'
            d_end = Hbin.bin(df, label='right')
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
        self.wrf = xr.merge((self.wrf, x.to_dataset('column')))

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

    def store(self):
        try:
            dt = pd.Timedelta(12, 'h')
            B = self.binned.loc[self.start+dt:].combine_first(self.loaded.binned)
        except:
            print('no loaded.binned')
            B = self.binned
        finally:
            B.to_hdf(config.CEAZAMet.raw_data, 'raw/binned/end/T2')

        try:
            wrf = self.wrf.combine_first(self.loaded.wrf)
            # need to treat XTIME separately because combine_first seems to drop unused coordinates
            wrf.coords['XTIME'] = self.wrf['XTIME'].combine_first(self.loaded.wrf['XTIME'])
        except:
            print('no loaded.wrf')
            wrf = self.wrf
        finally:
            wrf.to_netcdf(self.file_name, mode='w')

    @classmethod
    def load(cls, var_code):
        nt = namedtuple('data', ['binned', 'raw', 'wrf'])
        nc = xr.open_dataset(cls.file_name)
        nc.close()
        with pd.HDFStore(config.CEAZAMet.raw_data) as S:
            node = S.get_node('raw/{}'.format(var_code))
            raw = {k: S[v._v_pathname] for k, v in node._v_children.items()}
            binned = S['raw/binned/end/T2']
        return nt(binned, raw, nc)

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


def update(overlap=1):
    from data import CEAZA
    f = CEAZA.Field()
    try:
        f.update('ta_c', raw=True)
    except CEAZA.FieldsUpToDate:
        print('CEAZAMet data are up to date')

    D = WRFR.load('ta_c')
    start = (D.binned.index[-1] - pd.Timedelta(overlap, 'd')).date()
    print('start: {}'.format(start))

    data = {k: v.loc[start:] for k, v in D.raw.items()}
    print('data length: {}'.format(len([d for d in data.values() if d is not None])))

    w = WRFR(data, loaded=D)
    for kwargs in config.WRFop.variables:
        w.stats(**kwargs)
    return w
    # w.store()


def compare_xr(a, b):
    a, b = xr.align(a.dropna('station', 'all'), b.dropna('station', 'all'))
    d = {}
    for k in set(a.data_vars).intersection(b.data_vars):
        x = a[k]
        ij = np.where(abs(x - b[k]) < 1e-2)
        d[k] = {x.dims[i]: np.unique(x[x.dims[i]][j]) for i, j in enumerate(ij)}
    return d

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
    return d
