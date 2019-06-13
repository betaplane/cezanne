"""
Updatable WRF validation statistics
-----------------------------------

This module envisages two use cases: immediate computation of errors of WRF simulations w.r.t. observed data, and maintance of a regularly updated dataset of errors.

The main class :class:`Validate` contains most of the necessary logic, and several helper classes carry out secondary functions such as appropriate hourly binning of the raw station data (:class:`Hbin`) or checks of the differences between two datasets (:func:`compare_df`, :func:`compare_xr`, :func:`compare_idx`).

Configurable parameters are currently loaded from the config module under the ``WRFop`` ('WRF operacional') class and attached as class attributes to :class:`Validate`. They comprise the filenames in which data is saved and the variables from both the met stations and the WRF output which are being validated. Adding other variables to be validated is as simple as adding another item to the :attr:`Validate.variables` :obj:`list` (and performing the direct computation for past times, so that they can be continuously 'updated').

The direct use case consists of calling :meth:`Validate.data` with a data :obj:`dict` (as obtained from :mod:`data.CEAZA.Field`)::

    from data import CEAZA, WRF
    from datetime import datetime

    f = CEAZA.Field()
    f.get_all('ta_c', from_date=datetime(2019,4,1), to_date=datetime(2019,5,1))

    w = WRF.Validate()
    w.stats(raw_dict=f.data)
    w.store() # or not, depending on needs


.. NOTE::

    # explain the binning in the middle as well, and the structure of the 'variables' list
    Interpolated WRF data (as produced by :meth:`Validate.wrf_files`) is **always** first written to file (with name given by :attr:`Validate.wrf_intp`) and then loaded again. If the data needed exists at that file location, it is **never** recreated in order to save computations.

    The other products of :class:`Validate` are not written to file unless the method :meth:`Validate.store` is called. These are:
    1) The binned CEAZAMet station data, which is written to the HDF file :attr:`data.CEAZA.Field.raw_data` under the key ``raw/binned/end`` (since only the data labeled at the end of the binning interval is currently saved).
    2) The computed errors of the WRF simulation w.r.t. the station data (i.e. WRF minus station), which is written to the netCDF file :attr:`Validate.wrf_err` with the main WRF variable name appended (i.e. the name given as key 'wrfout_var' in the :obj:`dicts <dict>` in :attr:`Validate.variables`. One file is written per item in the :attr:`Validate.variables` list.

.. WARNING::

    This module currently only allows the validation of WRF output against 'raw' data from :mod:`~data.CEAZA` and expects the existence of accumulated (mean, minimum, maximum) WRF output files for the screen-level variables. The reason for this is that the errors are indeed smaller if the station data are binned to precisely the same time interval covered by the WRF simulations, and the accumulated statistics are used. The WRF namelist variables we currently set to turn on the output of these files are::

        &time_control:
            output_diagnostics = 1
            auxhist3_outname = “wrfxtrm_d<domain>_<date>”
            auxhist3_interval = 180,  60,   60
            frames_per_auxhist3 = 1000, 1000, 1000
            io_form_auxhist3 = 2

"""
from helpers import config
from glob import glob
from datetime import datetime, timedelta
from data.interpolate import BilinearInterpolator
from importlib import import_module
from collections import namedtuple
from functools import reduce, partial
from tqdm import tqdm
from copy import deepcopy
import xarray as xr
import pandas as pd
import numpy as np
import os, re, logging


def align_times(wrf, df, level=None):
    """Align CEAZAMet weather station data in the form of a :class:`~pandas.DataFrame` with WRF output concatenated along 2 temporal dimensions and interpolated in space to the station locations.

    The specific WRF output concatenation format is described in more detail in the docstring of :class:`Validate`; in short, one dimension named ``start`` refers to the start time of a simulation, whereas a second dimension ``Time`` refers to the (integer) write-out time step of the simulation. The write-out interval is assumed to be one hour. The station data will be appropriately multiplicated and folded in order to allow the subsequent extraction of various 'forecast lead times'. The :class:`~pandas.DataFrame` containing station data is expected to be of the shape and column structure returned by :class:`Hbin`.

    This function is used in method :meth:`Validate._stats_by_sensor`, which in turn is called by :meth:`Validate.stats`.

    :param wrf: WRF simulation output interpolated to station locations, as prepared by :meth:`Validate.wrf_files` (which in turn utilizes :class:`data.interpolate.BilinearInterpolator`).
    :type wrf: :class:`xarray.Dataset`
    :param df: CEAZAMet single-station dataset binned to hourly intervals by :class:`Hbin`
    :type df: :class:`pandas.DataFrame`
    :param level: If not ``None``, return a DataArray with only this level from ``df`` as index 'columns' - otherwise, copy the columns from ``df``.
    :returns: station data aligned in 2 temporal dimensions with the input ``wrf`` dataset
    :rtype: :class:`xarray.DataArray`

    """
    # necessary because some timestamps seem to be slightly off-round hours
    xt = wrf.XTIME.stack(t=('start', 'Time'))
    xt = xr.DataArray(pd.Series(xt.values).dt.round('h'), coords=xt.coords).unstack('t')
    idx = np.vstack(df.index.get_indexer(xt.sel(start=s)) for s in wrf.start)
    columns = df.columns if level is None else df.columns.get_level_values(level)
    return xr.DataArray(np.stack([np.where(idx>=0, df[c].values[idx].squeeze(), np.nan) for c in df.columns], 2),
                        coords = [wrf.coords['start'], wrf.coords['Time'], ('columns', columns)])



# NOTE: in the git repo there are some older attempts at binning which don't use the pandas 'resample'
# method. In case Hbin is ultimately unsuccesful, I can go back to those.

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
        """FIXME! briefly describe function

        :param cls: 
        :param df: 
        :param label: 
        :param start: 
        :returns: 
        :rtype: 

        """
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
        x.columns = df.columns.set_levels(x.columns, level='aggr')
        return x if start is None else x.loc[start:]

class Validate(config.WRFop):
    utc_delta = pd.Timedelta(-4, 'h')
    update_overlap = pd.Timedelta(1, 'd')
    sims_delta = pd.Timedelta(1, 'd')
    time_dim_coord = 'XTIME'
    wrf_dim_order = ('start', 'station', 'Time')
    K = 273.15

    def __init__(self, data=None, copy=None):
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG)
        if copy is not None:
            self.__dict__ = copy.__dict__
            return

        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            self.flds = S['fields']
            # always interpolate to all historically active stations - since, why not
            self.sta = reduce(lambda i, j: S[i].combine_first(S[j]),
                              sorted([k for k in S.keys() if re.search('station', k)], reverse=True))

        if False:
            if data is None:
                # when updating, we use two different starting points for station data and WRF files
                # (b/c WRF files are not subject to changes once written out, unlike the stations db)
                # NOTE: my binning will cause typically 2 hourly timesteps at the end/beginning to be sensitive to new data
                with pd.HDFStore(config.CEAZAMet.raw_data) as S:
                    t = S.select('raw/binned/end/{}'.format(var_code), start=-1).index[0]
                    self.start_bin = t - self.update_overlap
                    raw = {k: S.select(k, where=self.start_bin.strftime('index>="%Y-%m-%dT%h:%m"'))
                           for k in S.get_node('raw/{}'.format(var_code))._v_children.items()}
                self.raw = {k: v for k, v in raw.items() if v.shape[0]>0}
                end = max([df.dropna().index.max() for df in self.raw.values()])
                print('start-wrf: {}; end-wrf: {}; start-binning: {}'.format(self.start, end, self.start_bin))
            else:
                data = {k: v for k, v in data.items() if (v is not None and v.shape[0]>0)}
                print('data with length > 0: {}'.format(len(data)))
                mi, ma = zip(*[(lambda x: (x.min(), x.max()))(df.dropna().index) for df in data.values()])
                self.start, end = min(mi), max(ma)
                self.raw = data

    def dirs(self, dirpat=None, raw_dict=None, start=None, end=None):
        assert not (raw_dict is not None and start is not None), "Specify only one of 'raw_dict' or 'start'"
        if raw_dict is not None:
            mi, ma = zip(*[(lambda x: (x.min(), x.max()))(df.dropna().index)
                           for df in raw_dict.values() if df is not None and df.shape[0]>0])
            start, end = min(mi), max(ma)

        if dirpat is None:
            dirpat = self.directory_re
        all_dirs = []
        for p in self.paths:
            try:
                all_dirs.extend([os.path.join(p, d) for d in os.listdir(p) if dirpat.search(d)])
            except PermissionError: raise # occasionally happens that I dont have permissions in a dir
            except: pass

        all_dirs = sorted(all_dirs, key=lambda s:s[-10:])
        # there's a list with simulations with errors in config_mod
        for s in self.skip:
            try:
                all_dirs.remove(s)
            except: pass
        all_times = [datetime.strptime(d[-10:], '%Y%m%d%H') for d in all_dirs]

        start_idx = None
        if os.path.isfile(self.wrf_intp):
            with xr.open_dataset(self.wrf_intp) as ds:
                start_idx = ds.indexes['start'].sort_values()
            # looping through the directories is time-consuming, so I set a flag here in case all is up to date
            if pd.DatetimeIndex(all_times).equals(start_idx):
                self._intp_done = True
                return []

        # TODO: possibly for future include a check here for whether a simulation has terminated or is running,
        # just in case
        def test(d, t):
            try:
                assert os.path.isdir(d)
                if start_idx is not None:
                    assert t not in start_idx
                if start is not None:
                    assert t >= start
                if end is not None:
                    assert t <= end
            except AssertionError: return False
            else: return True

        dt = list(zip(all_dirs, all_times))
        dirs = [d for d, t in dt if test(d, t)]

        if raw_dict is not None:
            # insert additional directories in front whose simulations overlap with data
            for d, t in dt[all_dirs.index(dirs[0])-1::-1]:
                try:
                    out = [os.path.join(d, f) for f in os.listdir(d) if self.wrfout_re.search(f)]
                    with xr.open_dataset(sorted(out)[-1]) as ds:
                        if ds[self.time_dim_coord].values.max() >= start.asm8 and t not in start_idx:
                            dirs.insert(0, d)
                        else: break
                except: pass


        return dirs

    # this method only writes the interpolated data to file
    # implement any chunking behavior here in the future if the size of data presents challenges
    def wrf_files(self, dirs):
        from functools import reduce
        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            sta = reduce(lambda i, j: S[i].combine_first(S[j]),
                         sorted([k for k in S.keys() if re.search('station', k)], reverse=True))

        out = [f for f in os.listdir(dirs[0]) if self.wrfout_re.search(f)]
        with xr.open_dataset(os.path.join(dirs[0], out[0])) as ds:
            itp = BilinearInterpolator(ds, stations=self.sta)

        wrfout_vars = [v['wrfout_var'] for v in self.variables]
        wrfxtrm_vars = [k for v in self.variables for k in list(v['wrfxtrm_vars'].keys())
                        if 'wrfxtrm_vars' in v]
        o, x = [], []
        for d in tqdm(dirs):
            dl = os.listdir(d)
            out = [os.path.join(d, f) for f in dl if self.wrfout_re.search(f)]
            if len(out) == 0:
                logging.info('directory {} has no files'.format(d))
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

            if len(wrfxtrm_vars) > 0:
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

        intp = xr.concat(o, 'start').transpose(*self.wrf_dim_order)
        if len(wrfxtrm_vars) > 0:
            intp = xr.merge((intp, xr.concat(x, 'start').transpose(*self.wrf_dim_order)))

        if os.path.isfile(self.wrf_intp):
            with xr.open_dataset(self.wrf_intp) as ds:
                intp = intp.combine_first(ds.load())

        intp.to_netcdf(self.wrf_intp)

    def _load_earlier(self, binned, set_end=False):
        # intended outcome: if a dict is passed to stats(), set_end is set to True,
        # otherwise it is assumed that it's an update and there will be no end set
        start = min([d.index.min() for d in binned])
        end = max([d.index.max() for d in binned])
        dirs = [] if getattr(self, '_intp_done', False) else self.dirs(start=start, end=end if set_end else None)

        if len(dirs) > 0:
            self.wrf_files(dirs)

        return start, end

    def _stats_by_sensor(self, station, wrfout_var, wrfxtrm_vars, key=None, df=None, start=None, prog=None):
        if df is None:
            with pd.HDFStore(config.CEAZAMet.raw_data) as S:
                if start is None:
                    df = S.select(key)
                else:
                    df = S.select(key, where=start.strftime('index>="%Y-%m-%dT%H:%M"'))
        if df.shape[0] == 0:
            if prog is not None:
                prog.update(1)
            return None
        start += self.update_overlap / 2
        d_mid = Hbin.bin(df, label='center', start=start).dropna(0, 'all')
        d_end = Hbin.bin(df, label='right', start=start).dropna(0, 'all')
        start, end = self._load_earlier([d_mid, d_end], df is not None)

        with xr.open_dataset(self.wrf_intp) as ds:
            # NOTE: slice with .sel() includes start, end (not with .isel())
            x = ds[wrfout_var + wrfxtrm_vars.keys()].sel(station=station, start=slice(start, end)).load()

        dm = align_times(x, d_mid, 'aggr').sel(columns='ave') + self.K
        dm['columns'] = 'point'
        de = align_times(x, d_end, 'aggr') + self.K
        X = xr.concat((
            xr.concat([x[k] - de.sel(columns=v) for k, v in wrfxtrm_vars.items()], 'columns'),
            x[wrfout_var] - dm
        ), 'columns')
        if prog is not None:
            prog.update(1)
        return X, d_end

    def stats(self, raw_dict=None):
        sta = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        R, B = {}, {}

        if raw_dict is None:
            var_dicts = deepcopy(self.variables)
            with pd.HDFStore(config.CEAZAMet.raw_data) as S:
                for var_dict in var_dicts:
                    vc = var_dict['ceazamet_field']
                    try:
                        t = S.select('raw/binned/end/{}'.format(vc), start=-1).index[0]
                        var_dict['start'] = t - self.update_overlap
                    except: pass
                    node = S.get_node('raw/{}'.format(vc))
                    var_dict['keys'] = [n._v_pathname for n in node._v_children.values()]

            for var_dict in var_dicts:
                field = var_dict.pop('ceazamet_field')
                keys = var_dict.pop('keys')
                sensor_codes = [k.split('/')[-1] for k in keys]
                stations = [sta[s] for s in sensor_codes]
                f = partial(self._stats_by_sensor, prog=tqdm(total=len(keys)), **var_dict)
                r, b = zip(*[x for x in [f(station=s, key=k) for s, k in zip(stations, keys)] if x is not None])
                R[field] = dict(zip(stations, r))
                B[field] = dict(zip(sensor_codes, b))

        else:
            ceazamet_fields = set(f for df in raw_dict.values() for f in df.columns.get_level_values('field'))

            for s, df in tqdm(raw_dict.items()):
                if (df is not None and df.shape[0]>0):
                    field = df.columns.get_level_values('field').unique().item()
                    var_dict = self.var_dict(field=field)
                    r, b = self._stats_by_sensor(sta[s], df=df, start=None, **var_dict)
                    try:
                        R[field][s], B[field][s] = r, b
                    except:
                        R[field], B[field] = {s: r}, {s: b}

        self.binned = {k: self.dict2frame(v) for k, v in B.items()}

        self.err = {}
        # NOTE: In the wrfxtrm files, the first data point of a simulation run is always 0!
        # nan it out here
        for k, v in R.items():
            x = xr.concat(v.values(), 'station')
            d = {
                'station': slice(None),
                'columns': x.indexes['columns'].get_indexer(['ave', 'max', 'min']),
                'start': slice(None),
                'Time': 0,
            }
            x[tuple([d[c] for c in x.dims])] = np.nan
            self.err[k] = x.to_dataset('columns').transpose(*self.wrf_dim_order)

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
    def split_rename(filename, suffix):
        n, e = os.path.splitext(filename)
        return '{}_{}{}'.format(n, suffix, e)

    @classmethod
    def nc_overwrite(cls, data, filename):
        # There's some weird issue with overwriting a previously-opened file in xarray / netCDF4
        # (no matter how much I try to close the dataset or use a 'with' context).
        # I can't reproduce it reliably, except *this* piece of code without the renaming shenanigans
        # *always* fails. Hence the renaming shenanigans.
        fn = cls.split_rename(filename, 'overwrite')
        data.to_netcdf(fn, mode='w')
        os.rename(fn, filename)

    # need to treat XTIME separately because combine_first seems to drop unused coordinates
    @staticmethod
    def combine_first(a, b):
        c = a.combine_first(b)
        c.coords['XTIME'] = a['XTIME'].combine_first(b['XTIME'])
        return c

    @classmethod
    def var_dict(cls, field=None):
        if field is not None:
            return [v for v in deepcopy(cls.variables) if v.pop('ceazamet_field')==field][0]

    def store(self):
        file_exists = os.path.isfile(config.CEAZAMet.raw_data)
        with pd.HDFStore(config.CEAZAMet.raw_data) as S:
            for field, b in self.binned.items():
                key = 'raw/binned/end/{}'.format(field)
                S[key] = b.combine_first(S[key]) if (file_exists and key in S) else b

        for field, err in self.err.items():
            wrfout = self.var_dict(field=field)['wrfout_var']
            filename = self.split_rename(self.wrf_err, wrfout)
            try:
                with xr.open_dataset(filename) as ds:
                    err = self.combine_first(err, ds.load())
            except: pass
            err.to_netcdf(filename, mode='w')
            # self.nc_overwrite(err, self.split_rename(self.wrf_err, wrfout))

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

    def map_plot(self, lead=1, coq=None):
        cpl = import_module('plots')
        plt = import_module('matplotlib.pyplot')
        gs = import_module('matplotlib.gridspec')
        if coq is None:
            coq = cpl.Coquimbo()
        fig = plt.figure(figsize=(15, 4))
        g = gs.GridSpec(1, 32, left=.05, right=.95)

        x = self.err.isel(Time=slice(lead*24, (lead+1)*24))
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
            dfa = cls.dataframe(data.intp, time)[sta[sensor_code]].resample('h').asfreq()
            dfb = cls.dataframe(data.err, time)[sta[sensor_code]].resample('h').asfreq()
        except: pass

        field = d.columns.get_level_values('field').unique().item()

        ax = fig.add_subplot(g[:2, 0])
        ax.set_title(sta[sensor_code])
        bx = fig.add_subplot(g[2, 0], sharex=ax)
        for i, (a, b) in enumerate(self.var_dict(field=field)['wrfxtrm'].items()):
            ax.plot(d.xs(a, 1, 'aggr'), color=colrs[i], label='obs {}'.format(a))
            try:
                ax.plot(dfa[a] - cls.K, color=colrs[i+3], label='WRF {}'.format(a))
                bx.plot(dfb[b], color=colrs[i+3], label='err {}'.format(a))
            except: pass

        start = min([df.dropna().index.min() for df in data.raw.values()])
        if prev is not None:
            # Dfa, Dfb = cls.dataframe(prev.intp, time), cls.dataframe(prev.err, time)
            dt = pd.Timedelta(2, 'd')
            x = prev.binned.xs(sensor_code, 1, 'sensor_code').xs('ave', 1, 'aggr')
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
        df = nat(nc.isel(Time=time).to_dataframe())
        df = df.reset_index().pivot('XTIME', 'station', list(nc.data_vars))
        return df.sort_index(1).swaplevel(axis=1)

    def check_overlap(self, other, aggr='ave'):
        a = other.xs(aggr, 1, 'aggr')
        b = self.binned.xs(aggr, 1, 'aggr')
        c = a.columns.get_level_values(0).intersection(b.columns.get_level_values(0))
        return pd.concat([(a[s] - b[s]).dropna() for s in c], 1, keys=c).T

# TODO: include logic to check if all variables are in existing wrf_intp file
# and if not, first produce a merged wrf_intp file
def update():
    from data import CEAZA
    # the CEAZA module takes care of updating *its* data (hopefully)
    f = CEAZA.Field()

    w = WRFR()
    for var_dict in config.WRFop.variables:
        try:
            f.update(var_dict('ceazamet_field'), raw=True)
        except CEAZA.FieldsUpToDate:
            print('CEAZAMet data are up to date')
        # this call simply uses the current WRFR.wrf_intp file to determine the start
    w.wrf_files(w.dirs())
    w.stats()
    return w
    # w.store()

def HDF_dict(filename, var_code):
    with pd.HDFStore(filename) as S:
        node = S.get_node('raw/{}'.format(var_code))
        return {k: S[v._v_pathname] for k, v in node._v_children.items()}

def compare_xr(a, b):
    eq = (a == b).sum(('Time', 'start')).to_dataframe()
    neq = (a != b) * np.isfinite(a-b)
    def f(x, k):
        xt, v = xr.broadcast(x['XTIME'], x[k])
        y = xt.values[v]
        return y.min() if len(y) > 0 else None
    t = {k: v for k, v in {k: f(neq, k) for k in neq.data_vars}.items() if v is not None}
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

def compare_idx(a, b, time):
    return (a.isnull().astype(int) - b.isnull()).isel(start=time).sum(('Time', 'start')).to_dataframe()

def append_netcdf(x, name, mode):
    # https://github.com/pydata/xarray/issues/1849
    for k in x.variables:
        if 'contiguous' in x[k].encoding:
            del x[k].encoding['contiguous']
    x.to_netcdf(name, mode, unlimited_dims=('start', 'Time'))

# https://github.com/pydata/xarray/issues/1672
