from glue.core import Data
import numpy as np
import xarray as xr
import pandas as pd
import helpers as hh
from WuRF.base import align_stations

def daily_err(sl, ds, T_sta):
    x = ds.isel(Time=sl)
    return (x - align_stations(x, T_sta)).dropna('station', 'all')

def pd2Data(name, df):
    return Data(name, **{k: v.values for k, v in df.reset_index().iteritems()})

T2m = xr.open_dataarray('/nfs/HPC/arno/data/T2MEAN.nc')
T_raw = hh.stationize(hh.read_hdf({'condor': '/nfs/HPC/arno/data/station_raw_binned_end.h5'}, 'ta_c').drop('10', 1, level='elev'), 'avg') + 273.15
err = daily_err(slice(16, 40), T2m, T_raw).mean('start').transpose('station', 'Time')
sta = pd.read_hdf(hh.config.Meta.file_name, 'stations')
dz = (sta['d03'] - sta['elev'].astype(float)).to_frame('dz')

df = pd.concat((dz, sta['d03_var_sso']), 1).loc[err.station].reset_index()
df.index.name = 'idx'

i = np.repeat(np.arange(err.shape[0]), err.shape[1]).reshape(err.shape)
dc.append(Data('err', e=err.values, idx=i))
dc.append(pd2Data('z', df))
