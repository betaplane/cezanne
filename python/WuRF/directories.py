from helpers import config, sta
from glob import glob
from datetime import datetime, timedelta
from data.interpolate import BilinearInterpolator
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

class Hbin:
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

        w, e = cls.weights(np.unique(df.index.minute), {'center': 30, 'right': 0}[label])
        a = ave.resample(loffset=loffs, **kwargs).apply(cls.ave, weights=w)
        b = ave.resample(loffset=loffs - pd.offsets.Hour(1), **kwargs).apply(cls.ave, weights=e)
        d = a.add(b, fill_value=0)
        ave = d['sum'] / d['weight']
        # apparently, reusing a resampler leads to unpredictble results
        mi = df.xs('min', 1, 'aggr').resample(loffset=loffs, **kwargs).min().iloc[:, 0]
        ma = df.xs('max', 1, 'aggr').resample(loffset=loffs, **kwargs).max().iloc[:, 0]
        return pd.concat((ave, mi, ma), 1, keys=['ave', 'min', 'max'])


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

    def __init__(self, data, dir_pat='c01', copy=None):
        dirpat = re.compile(dir_pat)
        self.wrfout_re, self.wrfxtrm_re = re.compile(self.wrfout_re), re.compile(self.wrfxtrm_re)

        if isinstance(data, pd.DataFrame):
            start, end = self._x.index[[0, -1]] - self.utc_delta
            self.sta = data.columns.get_level_values('station')
        else:
            start = min([df.index[0] for df in data.values()])
            end = max([df.index[-1] for df in data.values()])
        self.data = data
        self.flds = pd.read_hdf(config.Meta.file_name, 'fields')

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
                assert t >= start
                assert t <= end
            except: return False
            else: return True

        self.dirs = sorted([d for d in dirs if test(d)], key=lambda s:s[-10:])

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

        # there's a list with simulations with errors in config_mod
        for s in self.skip:
            try:
                self.dirs.remove(s)
            except: pass

        if copy is not None:
            self.wrf = copy.wrf
            self.xtrm = copy.xtrm

    def wrf_files(self, wrfout_vars, wrfxtrm_vars=None, stations=sta):
        out = [f for f in os.listdir(self.dirs[0]) if self.wrfout_re.search(f)]
        with xr.open_dataset(os.path.join(self.dirs[0], out[0])) as ds:
            itp = BilinearInterpolator(ds, stations=stations)

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
                xi.coords['start'] = t.min()
                x.append(xi)

        self.wrf = xr.concat(o, 'start')
        if wrfxtrm_vars is not None:
            self.xtrm = xr.concat(x, 'start')


    def stats(self):
        # self.wrf_files(['T2'], ['T2MEAN', 'T2MIN', 'T2MAX'])

        sta = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        t = np.unique(self.wrf[self.time_dim_coord])
        X, self.D = {}, {}
        for s, df in tqdm(self.data.items()):
            d_mid = Hbin.bin(df, label='center')
            dm = align_times(self.wrf, d_mid).sel(column='ave') + self.K
            dm['column'] = 'point'
            d_end = align_times(self.xtrm, Hbin.bin(df, label='right')) + self.K
            w = self.wrf.sel(station=sta[s])
            x = self.xtrm.sel(station=sta[s])
            X[sta[s]] = xr.concat((
                x['T2MEAN'] - d_end.sel(column='ave'),
                x['T2MAX'] - d_end.sel(column='max'),
                x['T2MIN'] - d_end.sel(column='min'),
                w['T2'] - dm
            ), 'column')
            self.D[s] = d_mid
        x = xr.concat(X.values(), 'station')
        # NOTE: In the wrfxtrm files, the first data point of a simulation run is always 0!
        # nan it out here
        d = {
            'station': slice(None),
            'column': x.indexes['column'].get_indexer(['ave', 'max', 'min']),
            'start': slice(None),
            'Time': 0,
        }
        x[tuple([d[c] for c in x.dims])] = np.nan
        self.wrf_err = x.to_dataset(name='T2')

    def store(self):
        names = ['station', 'field', 'sensor_code', 'elev']
        sta = pd.read_hdf(config.Meta.file_name, 'stations')
        d = self.flds['sensor_code'].reset_index().join(sta, on='station')
        def df(k, v):
            df = v.copy()
            s = d[d['sensor_code']==k][names].values[0]
            df.columns = pd.MultiIndex.from_tuples([np.r_[s, [c]] for c in v.columns])
            return df
        D = pd.concat([df(k, v) for k, v in self.D.items()])
        with pd.HDFStore(config.Field.raw_data) as S:
            try:
                d = S['raw/binned/T2']
                D = D.combine_first(d)
            except: pass
            S['raw/binned/T2'] = D

        try:
            with xr.open_dataset(self.file_name) as ds:
                wrf = self.wrf.combine_first(ds)
        except:
            wrf = self.wrf

        wrf.to_netcdf(self.file_name)

def update(overlap=1):
    from data import CEAZA
    data = pd.read_hdf(config.Field.raw_data, 'raw/binned/T2')
    start = (data.index[-1] - pd.Timedelta(overlap, 'd')).date()
    f = CEAZA.Field()
    f.get_all('ta_c', from_date=start, raw=True)

    w = WRFR(f.data)
    w.stats()
    w.store()
