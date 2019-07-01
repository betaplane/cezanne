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

    * figure out what would need to be done to run this daily
    * implement a check of binned data against hourly (non-raw) from CEAZAMet
    * sequential writing of the netcdf file (at this point not possible with xarray, only netCDF4)

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
def log():
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
                # df = df.copy()
                # df.columns = df.columns.get_level_values('aggr')
                return df if start is None else df.loc[start:]
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

# class SensorFilter:
#     def __init__(self, field):
#         self.filter = getattr(self, field, lambda x: True)
#         flds = pd.read_hdf(config.CEAZAMet.meta_data, 'fields').xs(field, level='field')
#         self.sensors = flds.reset_index().set_index('sensor_code')
#         self.field = field

#     def ta_c(self, key):
#         k = key.split('/')[-1]
#         s = self.sensors.loc[k]
#         st = self.sensors[self.sensors['station']==s['station']]
#         if st.shape[0] > 1:
#             if abs(st['elev'] - 2).idxmin() != k:
#                 return False
#         return True


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
            self.sta = reduce(lambda i, j: S[i].combine_first(S[j]),
                              sorted([k for k in S.keys() if re.search('station', k)], reverse=True))

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
                                C.append(Hbin.bin(df, label='center', start=start).dropna(0, 'all'))
                                logger.info('{}, center: {}, start: {}'.format(f, k, start))
                            if 'end' in ce:
                                E.append(Hbin.bin(df, label='right', start=start).dropna(0, 'all'))
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

    def _stats_by_sensor(self, station, wrfout_var, wrfxtrm_vars, key=None, df=None, start=None, prog=None):
        logger.info('stats-by-sensor: key {}; wrf {}; start {}; df {}'.format(key, wrfout_var, start, df is not None))
        if df is None:
            field = key.split('/')[-1]
            with pd.HDFStore(config.CEAZAMet.raw_data) as S:
                if start is None:
                    df = S.select(key)
                else:
                    df = S.select(key, where=start.strftime('index>="%Y-%m-%dT%H:%M"'))
        else:
            field = df.columns.get_level_values('field').unique().items()
        if df.shape[0] == 0:
            if prog is not None:
                prog.update(1)
            return None
        if start is not None:
            start += self.update_overlap / 2
        d_mid = Hbin.bin(df, label='center', start=start).dropna(0, 'all')
        d_df_end = Hbin.bin(df, label='right', start=start).dropna(0, 'all')

        start = min([d.index.min() for d in [d_mid, d_end]])
        end = max([d.index.max() for d in [d_mid, d_end]])
        dirs = [] if getattr(self, '_intp_done', False) else self.dirs(start=start, end=None if df is None else end)
        if len(dirs) > 0:
            self.wrf_files(dirs)

        if transform_vars is not None:
            raise Exception('not implemented yet')
            # variables = transform_vars['wrfout'], transform_vars['wrfxtrm']
        else:
            variables = [wrfout_var] + list(wrfxtrm_vars.keys())
        # NOTE: slice with .sel() includes start, end (not with .isel())
        # I keep the 'start' variable at the UTC time stamps for now, since this corresponds to the directory names
        # but that means I need to reverse-adjust the local timestamps from the station data
        # (but not for XTIME, which is adjusted to start with!!!!!)
        with xr.open_dataset(self.wrf_intp) as ds:
            i, _ = np.where(ds[self.time_coord] >= np.datetime64(start))
            x = ds[variables].sel(station=station, start=slice(None, end-self.utc_delta))
            x = x.isel(start=slice(min(i), None)).load()

        f, g = Transforms.get(field)
        if f is not None:
            d_mid, d_end = f(d_mid), f(d_end)

        dm = align_times(x, d_mid, 'aggr').sel(columns='ave')
        dm['columns'] = 'point'
        de = align_times(x, d_end, 'aggr')
        X = xr.concat((
            xr.concat([x[k] - de.sel(columns=v) for k, v in wrfxtrm_vars.items()], 'columns'),
            x[wrfout_var] - dm
        ), 'columns')
        if prog is not None:
            prog.update(1)
        return X, d_end

    def _stats_old(self, raw_dict=None):
        sta = {k: v for (v, _), k in self.flds['sensor_code'].items()}
        R, B = {}, {}

        n = 0
        if raw_dict is None:
            obs = deepcopy(self.obs)
            var_dicts = deepcopy(self.variables)
            with pd.HDFStore(config.CEAZAMet.raw_data) as S:
                for k, vdict in obs.items():
                    sf = SensorFilter(k)
                    try:
                        t = S.select('raw/binned/end/{}'.format(k), start=-1).index[0]
                        vdict['start'] = t - self.update_overlap
                    except: pass
                    node = S.get_node('raw/{}'.format(k))
                    keys = [n._v_pathname for n in node._v_children.values()]
                    keys = [s for s in keys if sf.filter(s)]
                    vdict['keys'] = keys
                    n += len(keys)

            prog = tqdm(total=n)
            for f, vdict in obs:
                keys = vdict.pop('keys')
                stations = [sta[k.split('/')[-1]] for k in keys]
                r, B[field] = zip(*[x for x in [self._stats_by_sensor(station=s, key=k, prog=prog, **vdict)
                                                for s, k in zip(stations, keys)] if x is not None])
                R[field] = dict(zip(stations, r))

        else:
            ceazamet_fields = set(f for df in raw_dict.values() for f in df.columns.get_level_values('field'))

            for s, df in tqdm(raw_dict.items()):
                if (df is not None and df.shape[0]>0):
                    field = df.columns.get_level_values('field').unique().item()
                    try:
                        assert sf.field == field
                    except:
                        sf = SensorFilter(field)
                    if sf.filter(s):
                        var_dict = self.var_dict(field=field)
                        r, b = self._stats_by_sensor(sta[s], df=df, start=None, **var_dict)
                        try:
                            R[field][s] = r
                            B[field].append(b)
                        except:
                            R[field] = {s: r}
                            B[field] = [b]

        self.binned = {k: pd.concat(v, 1) for k, v in B.items()}

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

    @staticmethod
    def split_rename(filename, suffix):
        n, e = os.path.splitext(filename)
        return '{}_{}{}'.format(n, suffix, e)

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

    def store(self, ceazamet_file=None, WRF_err_file=None):
        filename = config.CEAZAMet.raw_data if ceazamet_file is None else ceazamet_file
        file_exists = os.path.isfile(filename)
        with pd.HDFStore(filename) as S:
            for field, b in self.binned.items():
                key = 'raw/binned/end/{}'.format(field)
                S[key] = b.combine_first(S[key]) if (file_exists and key in S) else b

        for field, err in self.err.items():
            wrfout = self.var_dict(field=field)['wrfout_var']
            filename = self.split_rename(self.wrf_err if WRF_err_file is None else WRF_err_file, wrfout)
            try:
                with xr.open_dataset(filename) as ds:
                    err = self.combine_first(err, ds.load())
            except: pass
            err.to_netcdf(filename, mode='w')
            # self.nc_overwrite(err, self.split_rename(self.wrf_err, wrfout))

    # @classmethod
    # def load(cls, var_code, limit=False):
    #     nt = namedtuple('data', ['binned', 'raw', 'intp', 'err'])
    #     l = int(limit)
    #     ds = xr.open_dataset(cls.wrf_intp)
    #     intp = ds.isel(start=slice(*[(None,), (-10, None)][l])).load()
    #     ds.close()
    #     try:
    #         ds = xr.open_dataset(cls.wrf_err)
    #         err = ds.isel(start=slice(*[(None,), (-10, None)][l])).load()
    #         ds.close()
    #     except:
    #         err = None
    #     with pd.HDFStore(config.CEAZAMet.raw_data) as S:
    #         node = S.get_node('raw/{}'.format(var_code))
    #         raw = {k: S[v._v_pathname].iloc[slice(*[(None,), (-1000, None)][l])]
    #                for k, v in node._v_children.items()}
    #         binned = S['raw/binned/end/{}'.format(var_code)].iloc[slice(*[(None,), (-100, None)][l])]
    #     return nt(binned, raw, intp, err)

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

    # def check_overlap(self, other, aggr='ave'):
    #     a = other.xs(aggr, 1, 'aggr')
    #     b = self.binned.xs(aggr, 1, 'aggr')
    #     c = a.columns.get_level_values(0).intersection(b.columns.get_level_values(0))
    #     return pd.concat([(a[s] - b[s]).dropna() for s in c], 1, keys=c).T

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


def append_netcdf(x, name, mode):
    # https://github.com/pydata/xarray/issues/1849
    for k in x.variables:
        if 'contiguous' in x[k].encoding:
            del x[k].encoding['contiguous']
    x.to_netcdf(name, mode, unlimited_dims=('start', 'Time'))

# https://github.com/pydata/xarray/issues/1672

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
            f.store_raw()

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Validate WRF simulations against CEAZAMet station records.')
    p.add_argument('command', action='store', choices=['daily', 'weekly'])
    p.add_argument('--from_date', action='store', type=pd.Timestamp, default=None)
    p.add_argument('--to_date', action='store', type=pd.Timestamp, default=None)

    c = p.parse_args()
    log()
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
            s = f.read().splitlines()
            last = max([datetime.strptime(l, 'daily @ %Y-%m-%d %H:%M:%S')
                        for l in s if l[:5]=='daily'])
        print(last)
    else:
        p.print_help()
