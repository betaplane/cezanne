#!/home/arno/Documents/code/conda/envs/intel/bin/python
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

.. TODO::

    * implement a check of binned data against hourly (non-raw) from CEAZAMet
    * to make binning more efficient, first split dfs according to datalogger interval changes
    * check how to enable adding back-data of new variable to the update process/files
    * sequential writing of the netcdf file (at this point not possible with xarray, only netCDF4):
        * https://github.com/pydata/xarray/issues/1849
        * https://github.com/pydata/xarray/issues/1672
        * see also function 'append_netcdf' in old git check-ins

"""
__package__ = 'data'
from helpers import config
from formulae import rh2w, w2rh
from glob import glob
from datetime import datetime, timedelta
from .interpolate import BilinearInterpolator
from importlib import import_module
from collections import namedtuple, Counter
from functools import reduce, partial
from tqdm import tqdm
from copy import deepcopy
import xarray as xr
import pandas as pd
import numpy as np
import os, re, logging

variables = [
    {
        'transf': 't2',
        'obs': ['ta_c'],
        'wrf': {
            'out': ['T2'],
            'xtrm': ['T2MEAN', 'T2MIN', 'T2MAX']
        }
    }, {
        'transf': 'q2',
        'obs': {
            'center': ['hr'],
            'end': ['hr', 'ta_c', 'pa_hpa', 'pa_kpa']
        },
        'wrf': {
            'out': ['Q2', 'T2', 'PSFC'],
            'xtrm': ['Q2MEAN', 'T2MEAN']
        }
    }
]

logger = logging.getLogger(__name__)
# this is supposed to get executed only once (in particular not if 'reload' is used) -
# otherwise each logged line is added to the file as many times as identical file handlers have been added
if len(logger.handlers) == 0:
    fh = logging.FileHandler(config.WRFop.log_file)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

def ceazamet_fields():
    F = {}
    for vdict in variables:
        obs = vdict['obs']
        try:
            for k, v in obs.items():
                F.update({f: F.get(f, set()).union({k}) for f in v})
        except:
            F.update({f: {'center', 'end'} for f in obs})
    return F

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

def csel(ds, **kwargs):
    x = ds.copy()
    idx = x.indexes['columns']
    for k, v in kwargs.items():
        i, idx = idx.get_loc_level(v, level=k)
        x = x.sel(columns=i)
    x = x.rename({'columns': 'station'})
    x.coords['station'] = ('station', idx.get_level_values('station'))
    return x

# NOTE: in the git repo there are some older attempts at binning, some with pandas.resample, some without
# In case hourly_bin is ultimately unsuccesful, I can go back to those.

def hourly_bin(df, label='right', start=None):
    x = pd.DataFrame(df.values, index=df.index, columns=df.columns.get_level_values('aggr'))
    assert 0 in x.index.minute, "not hour-aligned"
    dt = np.diff(x.index).min().astype('timedelta64[s]').astype(int)
    x = x.shift(freq=pd.offsets.Second(-dt/2))

    if label=='center':
        x = x.shift(freq=pd.offsets.Minute(30))
    shift = pd.offsets.Hour(1 if label=='right' else 0)

    # create a pivot table with date+hour as index and minutes as columns,
    # shifted in accordance with the 'label' argument
    idx = pd.DatetimeIndex(x.index.date)+pd.TimedeltaIndex(x.index.hour,'h')
    X = x.assign(second=x.index.second).assign(idx=idx).pivot(index='idx', columns='second')
    return X
    def ave(row):
        x = row.dropna()
        try: return np.trapz(x.values, x=x.index) / np.diff(x.index[[0, -1]]).item()
        except: return np.nan

    # far more efficient to apply the 'raw' np.trapz function to those hours that don't have missing values
    a = X['ave']
    i = a.isnull().sum(1)==0
    a = pd.concat((
        a.loc[i].apply(np.trapz, 1, dx=dt, raw=True) / np.diff(a.columns[[0, -1]]).item(),
        a.loc[~i].apply(ave, 1)
    )).sort_index()

    mi = X['min'].apply(np.nanmin, 1, raw=True)
    ma = X['max'].apply(np.nanmax, 1, raw=True)
    z = pd.concat((a, mi, ma), 1, keys=['ave', 'min', 'max']).shift(freq=shift)
    z.columns = df.columns.set_levels(z.columns, level='aggr')
    return z if start is None else z.loc[start:]


class Transforms:
    @staticmethod
    def t2(obs_cen, obs_end, wrf_out, wrf_xtrm):
        ds_cen = csel(align_times(wrf_out, obs_cen + 273.15), aggr='ave')
        out = (wrf_out - ds_cen).expand_dims({'aggr': ['point']})
        ds_end = align_times(wrf_xtrm, obs_end + 273.15)
        xtrm = xr.concat(
            [wrf_xtrm[i] - csel(ds_end, aggr=j)
             for i, j in [('T2MEAN', 'ave'), ('T2MIN', 'min'), ('T2MAX', 'max')]],
            pd.Index(['ave', 'min', 'max'], name='aggr')
        ).to_dataset(name='T2').isel(Time=slice(1, None)) # nixing out Time 0 for xtrm values
        return xr.concat((out, xtrm), 'aggr')

    @staticmethod
    def q2(obs_cen, obs_end, wrf_out, wrf_xtrm):
        def k2hpa(df):
            try:
                i, _ = df.columns.get_loc_level('pa_kpa', level='field')
                x = df.iloc[:, i] * 10
                idx = x.columns
                idx = idx.set_levels([k if k!='pa_kpa' else 'pa_hpa'
                                      for k in idx.levels[idx.names.index('field')]], level='field')
                x.columns = idx
                df = pd.concat((df.iloc[:, ~i], x), 1)
            except: pass
            finally: return df
        obs_cen, obs_end = k2hpa(obs_cen), k2hpa(obs_end)

        def dl(x):
            x.columns = x.columns.droplevel(['sensor_code', 'elev'])
            return x
        HR = dl(obs_end.xs('hr', 1, 'field'))
        T = dl(obs_end.xs('ta_c', 1, 'field')) + 273.15
        p = dl(obs_end.xs('pa_hpa', 1, 'field')) * 100

        s = reduce(
            lambda x, y: x.intersection(y),
            [x.columns.get_level_values('station').unique() for x in [HR, T, p]]
        )
        q_end = csel(align_times(wrf_xtrm, rh2w(HR[s], T[s], p[s])), aggr='ave')

        # because xtrm values are 0 at 'Time' 0
        q = (wrf_xtrm['Q2MEAN'] - q_end).isel(Time=slice(1, None)).expand_dims({'aggr': ['ave']})

        rh = w2rh(wrf_out['Q2'], wrf_out['T2'], wrf_out['PSFC'])
        obs = csel(align_times(rh, dl(obs_cen.xs('hr', 1, 'field'))), aggr='ave')
        rp = (rh - obs).expand_dims({'aggr': ['point']})

        p = wrf_out['PSFC']
        p_intp = p.interp({'Time': p['Time'][:-1]+.5})
        p_intp.coords['Time'] = ('Time', np.arange(1, p['Time'].size))
        p_intp.coords['XTIME'] = (p['XTIME'].dims, p['XTIME'].isel(Time=slice(1, None)))
        T = wrf_xtrm['T2MEAN'].isel(Time=slice(1, None))
        q = wrf_xtrm['Q2MEAN'].isel(Time=slice(1, None))
        rh = w2rh(q, T, p_intp)
        hr = csel(align_times(rh, HR), aggr='ave')
        ra = (rh - hr).expand_dims({'aggr': ['ave']})
        return xr.merge((q.to_dataset(name='Q2'), xr.concat((rp, ra), 'aggr').to_dataset(name='RH')))

class SensorFilter:
    @staticmethod
    def elev(df, elev=2):
        a = df.columns.to_frame().reset_index(drop=True).drop('aggr', axis=1).drop_duplicates()
        for f in a['field'].unique():
            b = a[a['field']==f]['station']
            for s in b[b.duplicated()]:
                c = a[a['station']==s]
                idx = abs(c['elev'].astype(float) - elev).idxmin()
                df.drop(a.loc[c.drop(idx).index]['sensor_code'], 1, level='sensor_code', inplace=True)

class Validate(config.WRFop):
    utc_delta = pd.Timedelta(-4, 'h')
    update_overlap = pd.Timedelta(1, 'd')
    time_coord = 'XTIME'
    wrf_dim_order = ('start', 'station', 'Time', 'config', 'GFS_resolution')
    resolution_re = re.compile('resolution', re.IGNORECASE)

    # call this only once per session if logging is desired
    # logging is a singleton, and even a new instance of Validate doesn't need to reconfigure the logger -
    # it would result in multiple handlers all logging to the same file (i.e. multiple entries for the same line)

    def __init__(self):
        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            self.flds = S['fields']
            # always interpolate to all historically active stations - since, why not
            self.sta = reduce(lambda i, j: i.combine_first(j),
                              map(S.get, sorted([k for k in S.keys() if re.search('stations', k)], reverse=True)))

    @classmethod
    def dirs_dict(cls, raw_dict, **kwargs):
        mi, ma = zip(*[(lambda x: (x.min(), x.max()))(df.dropna().index)
                       for df in raw_dict.values() if df is not None and df.shape[0]>0])
        return cls.dirs(start=min(mi), end=max(ma), **kwargs)

    @classmethod
    def dirs(cls, start=None, end=None, var=None, add_at_start=False, dirpat='c01|c05'):
        # start and end in local time
        dp = re.compile(dirpat)
        all_dirs = []
        for p in cls.paths:
            try:
                all_dirs.extend([os.path.join(p, d) for d in os.listdir(p) if dp.search(d)])
            except PermissionError: raise # occasionally happens that I dont have permissions in a dir
            except: pass

        all_dirs = sorted([d for d in all_dirs if os.path.isdir(d)], key=lambda s:s[-10:])
        # there's a list with simulations with errors in config_mod
        for s in cls.skip:
            try:
                all_dirs.remove(s)
            except: pass
        all_times = [datetime.strptime(d[-10:], '%Y%m%d%H') for d in all_dirs]

        start_idx = None
        if os.path.isfile(cls.wrf_intp):
            with xr.open_dataset(cls.wrf_intp) as ds:
                if var is not None:
                    try:
                        start_idx = ds[var].dropna('start', 'all').indexes['start'].sort_values()
                    except: pass
                else:
                    start_idx = ds.indexes['start'].sort_values()

        start_utc = None if start is None else start - cls.utc_delta
        end_utc = None if end is None else end - cls.utc_delta
        def test(d, t):
            try:
                if start_idx is not None:
                    assert t not in start_idx
                if start_utc is not None:
                    assert t >= start_utc
                if end_utc is not None:
                    assert t <= end_utc
            except AssertionError: return False
            else: return True

        dt = list(zip(all_dirs, all_times))
        dirs = [d for d, t in dt if test(d, t)]

        # insert additional directories in front whose simulations overlap with data
        if add_at_start:
            i = pd.DatetimeIndex(all_times).get_loc(start_utc, method='nearest')
            i -= int(all_times[i]>start_utc)

            for d, t in dt[i::-1]:
                try:
                    out = [os.path.join(d, f) for f in os.listdir(d) if cls.wrfout_re.search(f)]
                    with xr.open_dataset(sorted(out)[-1]) as ds:
                        # XTIME is on local time, as is start
                        if ds[cls.time_coord].values.max() >= start.asm8 and t not in start_idx:
                            dirs.insert(0, d)
                        else: break
                except: pass
        return dirs

    def interpolate_wrf(self, dirs):
        logger.info('\ninterpolate_wrf\n---------')
        if len(dirs) == 0:
            logger.info('no new directories')
            return
        wrfout_vars = list(set([i for j in [v['wrf'].get('out', []) for v in variables] for i in j]))
        wrfxtrm_vars = list(set([i for j in [v['wrf'].get('xtrm', []) for v in variables] for i in j]))
        self.wrf_files(dirs, wrfout_vars, wrfxtrm_vars)

    def wrf_files(self, dirs, wrfout_vars, wrfxtrm_vars):
        # this method only writes the interpolated data to file
        # implement any chunking behavior here in the future if the size of data presents challenges
        with pd.HDFStore(config.CEAZAMet.meta_data) as S:
            sta = reduce(lambda i, j: S[i].combine_first(S[j]),
                         sorted([k for k in S.keys() if re.search('station', k)], reverse=True))

        out = [f for f in os.listdir(dirs[0]) if self.wrfout_re.search(f)]
        with xr.open_dataset(os.path.join(dirs[0], out[0])) as ds:
            itp = BilinearInterpolator(ds, stations=self.sta)

        o, x = [], []
        for d in tqdm(dirs):
            dirpat = os.path.split(d)[1].split('_')[0]
            try:
                dl = os.listdir(d)
            except PermissionError:
                logger.error('{} permission denied'.format(d))
                continue
            try:
                # if this file doesn't exist yet, the simulation is still running
                res = [f for f in dl if self.resolution_re.search(f)][0].split('_')[1]
            except:
                logger.error('{} still running'.format(d))
                continue
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

            t = oi.coords[self.time_coord]
            oi.coords['start'] = t.min()
            t += self.utc_delta
            oi.coords[self.time_coord] = t
            oi = oi.expand_dims({'config': [dirpat]}).expand_dims({'GFS_resolution': [res]})
            o.append(oi)
            logger.info('{}: interpolated {}'.format(d, wrfout_vars))

            if len(wrfxtrm_vars) > 0:
                xtrm = [os.path.join(d, f) for f in dl if self.wrfxtrm_re.search(f)]
                if len(xtrm) > 1:
                    with xr.open_mfdataset(xtrm) as ds:
                        xi = ds[wrfxtrm_vars].apply(itp.xarray)
                else:
                    with xr.open_dataset(xtrm[0]) as ds:
                        xi = ds[wrfxtrm_vars].apply(itp.xarray)

                xi.coords[self.time_coord] = t
                xi.coords['start'] = oi.coords['start']
                xi = xi.expand_dims({'config': [dirpat]}).expand_dims({'GFS_resolution': [res]})
                x.append(xi)
                logger.info('{}: interpolated {}'.format(d, wrfxtrm_vars))

        intp = xr.concat(o, 'start').transpose(*self.wrf_dim_order)
        if len(wrfxtrm_vars) > 0:
            intp = xr.merge((intp, xr.concat(x, 'start').transpose(*self.wrf_dim_order)))

        if os.path.isfile(self.wrf_intp):
            with xr.open_dataset(self.wrf_intp) as ds:
                intp = self.combine_first(intp, ds.load())

        intp.transpose(*self.wrf_dim_order).to_netcdf(self.wrf_intp)

    @classmethod
    def bin(cls, fields=ceazamet_fields(), start=None):
        logger.info('\nbin\n---')
        CD, ED = {}, {}
        with pd.HDFStore(config.CEAZAMet.raw_data) as S:
            N = sum([getattr(S.get_node('raw/{}'.format(f)), '_v_nchildren', 0) for f in fields.keys()])
            prog = tqdm(total=N)
            for f, ce in fields.items():
                C, E = [], []
                key = 'raw/binned/end/{}'.format(f)
                if start is None:
                    try:
                        start = S.select(key, start=-1).index[0]
                    except: pass
                if start is not None:
                    if datetime.now() - start < cls.update_overlap:
                        logger.info('binned up to date')
                        return
                    start -= cls.update_overlap
                node = S.get_node('raw/{}'.format(f))
                if node is not None:
                    for k, v in node._v_children.items():
                        if start is None:
                            df = S.select(v._v_pathname)
                        else:
                            df = S.select(v._v_pathname, where=start.strftime('index>="%Y-%m-%dT%H:%M"'))
                            start += cls.update_overlap / 2
                        if not df.empty:
                            if 'center' in ce:
                                C.append(hourly_bin(df, label='center', start=start).dropna(0, 'all'))
                                logger.info('{}, center: {}, start: {}'.format(f, k, start))
                            if 'end' in ce:
                                E.append(hourly_bin(df, label='right', start=start).dropna(0, 'all'))
                                logger.info('{}, end: {}'.format(f, k, start))
                        prog.update(1)
                    if len(E) > 0:
                        ED[f] = pd.concat(E, 1).combine_first(S[key]) if key in S else pd.concat(E, 1)
                    if len(C) > 0:
                        key = 'raw/binned/center/{}'.format(f)
                        CD[f] = pd.concat(C, 1).combine_first(S[key]) if key in S else pd.concat(C, 1)

        for f, x in ED.items():
            x.to_hdf(config.CEAZAMet.raw_data, key='raw/binned/end/{}'.format(f), mode='a', format='table')
        for f, x in CD.items():
            x.to_hdf(config.CEAZAMet.raw_data, key='raw/binned/center/{}'.format(f), mode='a', format='table')

    @staticmethod
    def bin_dict(raw_dict):
        fields = ceazamet_fields()
        prog = tqdm(total=len(raw_dict))
        C, E = [], []
        for k, df in raw_dict.items():
            field = df.columns.get_level_values('field').unique().item()
            ce = fields[field]
            if not df.empty:
                if 'center' in ce:
                    b = Hbin.bin(df, label='center', start=start).dropna(0, 'all')
                    C.append(b)
                if 'end' in ce:
                    b = Hbin.bin(df, label='right', start=start).dropna(0, 'all')
                    E.append(b)
            prog.update(1)
        C = xr.concat(C, 1) if len(C) > 0 else None
        E = xr.concat(E, 1) if len(E) > 0 else None
        return C, E

    def _stats(self, obs, wrf, transf, start=None, end=None):
        s = '\nstats start: {}, end: {}'.format(start, end).strip()
        logger.info('{}\n{}'.format(s, ''.join(['-' for _ in range(len(s))])))

        if start is not None:
            where = start.strftime('index>="%Y-%m-%dT%H:%M"')
            if end is not None:
                where = '&'.join((where, end.strftime('index<="%Y-%m-%dT%H:%M"')))
        elif end is not None:
            where = end.strftime('index<="%Y-%m-%dT%H:%M"')
        else:
            where = None

        df_cen, df_end, out, xtrm = None, None, None, None
        with pd.HDFStore(config.CEAZAMet.raw_data) as S:
            def f(fields, label):
                return ['raw/binned/{}/{}'.format(label, f) for f in fields]
            if isinstance(obs, list):
                # if the fields/transform info is the same for centered or end-labeled binning, I don't replicate it
                # for 'center' and 'end' items in the top-level dict (e.g. 'ta_c')
                df_cen = pd.concat([S.select(k, where=where) for k in f(obs, 'center') if k in S], 1)
                df_end = pd.concat([S.select(k, where=where) for k in f(obs, 'end') if k in S], 1)
                logger.info('obs: {}'.format(obs))
            else:
                # in the case of e.g. 'hr'
                if 'center' in obs:
                    df_cen = pd.concat([S.select(k, where=where) for k in f(obs['center'], 'center') if k in S], 1)
                if 'end' in obs:
                    df_end = pd.concat([S.select(k, where=where) for k in f(obs['end'], 'end') if k in S], 1)
                logger.info('obs: {}'.format(list(set(obs['center']).union(obs['end']))))

        logger.info('wrf: {}'.format(set(wrf.get('out', [])).union(wrf.get('xtrm', []))))

        with xr.open_dataset(self.wrf_intp) as ds:
            t = ds[self.time_coord].transpose('start', 'Time')
            # XTIME/time_coord is adjusted to local time, so no need to correct start, end
            if start is not None:
                i, _ = np.where(t >= np.datetime64(start))
                i = set(i)
            if end is not None:
                j, _ = np.where(t >= np.datetime64(end))
                try:
                    i = i.intersection(j)
                except: i = set(j)
            if 'out' in wrf:
                out = ds[wrf['out']].sel(station=df_cen.columns.get_level_values('station').unique())
                try:
                    out = out.isel(start=i).load()
                except: out.load()
            if 'xtrm' in wrf:
                d = wrf['xtrm']
                xtrm = ds[wrf['xtrm']].sel(station=df_end.columns.get_level_values('station').unique())
                try:
                    xtrm = xtrm.isel(start=i).load()
                except: xtrm.load()

        SensorFilter.elev(df_cen)
        SensorFilter.elev(df_end)
        return getattr(Transforms, transf)(df_cen, df_end, out, xtrm)

    def stats(self, start=None, end=None):
        err = xr.merge([self._stats(start=start, end=end, **vdict) for vdict in variables])
        if os.path.isfile(self.wrf_err):
            with xr.open_dataset(self.wrf_err) as ds:
                err = self.combine_first(err, ds.load())
        err.to_netcdf(self.wrf_err, mode='w')

    # need to treat XTIME separately because combine_first seems to drop unused coordinates
    @staticmethod
    def combine_first(a, b):
        c = a.combine_first(b)
        c.coords['XTIME'] = a['XTIME'].combine_first(b['XTIME'])
        return c

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
        for i, (a, b) in enumerate(self.var_dict(field=field)['wrfxtrm_vars'].items()):
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

def HDF_dict(filename, var_code):
    with pd.HDFStore(filename) as S:
        node = S.get_node('raw/{}'.format(var_code))
        return {k: S[v._v_pathname] for k, v in node._v_children.items()}

def compare_xr(a, b):
    eq = (a == b).sum(('Time', 'start')).to_dataframe()
    neq = (a != b) * np.isfinite(a-b)
    def f(x, k):
        xt, v = xr.broadcast(a['XTIME'], x[k])
        y = xt.values[v.values]
        return y.min() if len(y) > 0 else None
    t = {k: v for k, v in {k: f(neq, k) for k in neq.data_vars}.items() if v is not None}
    neq = neq.sum(('Time', 'start')).to_dataframe()
    return pd.concat((neq, eq), 1, keys=['neq', 'eq']).swaplevel(axis=1).sort_index(1), t

def compare_df(a, b):
    pass

def compare_dicts(a, b):
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

def compare_idx(a, b):
     x = (a.isnull().astype(int) - b.isnull()).to_array()
     j = [x[i] for i in zip(*np.where(x))]
     if len(j)==0:
         print('No index differences detected.')
     else:
         return xr.concat(j).to_dataframe('null').set_index('station').sort_index()
     # s = df.sum(('Time', 'start')).isel(start=time).to_dataframe()

def update_ceazamet(**kwargs):
    logger.info('\nupdate_ceazamet\n---------------')
    from . import CEAZA
    with pd.HDFStore(config.CEAZAMet.raw_data) as S:
        node = S.get_node('raw')
        fields = node._v_children.keys() if node is not None else {}
    for k in ceazamet_fields().keys():
        f = CEAZA.Field()
        if k in fields:
            logger.info('updating {}'.format(k))
            try:
                f.update(k, **kwargs)
            except CEAZA.FieldsUpToDate:
                logger.info('CEAZAMet data up to date')
        else:
            logger.info('getting {} for first time'.format(k))
            f.get_all(k, raw=True, **kwargs)
            # f.store_raw()

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Validate WRF simulations against CEAZAMet station records.')
    p.add_argument('command', action='store', choices=['daily', 'weekly'])
    p.add_argument('--from_date', action='store', type=pd.Timestamp, default=None)
    p.add_argument('--to_date', action='store', type=pd.Timestamp, default=None)

    c = p.parse_args()
    s = '{} @ {}'.format(c.command, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info('\n{}\n{}'.format(s, ''.join(['*' for _ in range(len(s))])))

    backstop = pd.Timestamp("2019-06-01")
    if c.from_date is not None:
        backstop = c.from_date

    w = Validate()
    if c.command == 'daily':
        update_ceazamet(from_date=backstop, to_date=c.to_date)
        w.interpolate_wrf(dirs=w.dirs(start=backstop, dirpat='c01|c05'))
        w.bin(start=c.from_date) # bin needs no 'backstop', it will bin from the right time if start is None

        test = True
        if c.from_date is None:
            with xr.open_dataset(w.wrf_intp) as ds:
                start = ds.indexes['start']
                if os.path.isfile(w.wrf_err):
                    with xr.open_dataset(w.wrf_err) as ds:
                        start = start.difference(ds.indexes['start'])
                        if len(start)>0:
                            start = start.min()
                        else:
                            logger.error('no interpolated fields whose errors have not yet been calculated')
                            test = False
        else:
            start = c.from_date
        if test:
            w.stats(start=start)

    elif c.command == 'weekly':
        with open(w.log_file) as f:
            if c.from_date is None:
                # read the last 'weekly' update time from the log file
                s = f.read().splitlines()
                start = max([datetime.strptime(l, 'weekly @ %Y-%m-%d %H:%M:%S')
                             for l in s if l[:6]=='weekly']) - w.update_overlap
            else:
                start = c.from_date
            update_ceazamet(from_date=start)
            w.bin(start=start)
            w.interpolate_wrf(dirs=w.dirs(start=start), dirpat='c01|c05')
            w.stats(start=start)
    else:
        p.print_help()
