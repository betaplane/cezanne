from xarray import Dataset
from numpy import arange

def sel_rename(x, *args):
    x = x.sel(bottom_top_stag=slice(*args)).rename({'bottom_top_stag': 'bottom_top'})
    x['bottom_top'] = ('bottom_top', arange(x.bottom_top.size))
    return x

def temp_geopotential(ds):
    T = (ds['T'] + 300.) * ((ds['P'] + ds['PB']) / 1e5) ** 0.286
    gp = (ds['PH'] + ds['PHB']) / 9.81
    gpi = (sel_rename(gp, None, gp.bottom_top_stag.size-1) + sel_rename(gp, 1, None)) / 2
    return Dataset({'T': T, 'GP': gpi})

def bottom_layers(ds):
    a = ds[['T', 'P', 'PB']].isel(bottom_top=[0, 1])
    T = (a['T'] + 300.) * ((a['P'] + a['PB']) / 1e5) ** 0.286
    b = ds[['PH', 'PHB']].isel(bottom_top_stag=slice(0, 3))
    gp = (b['PH'] + b['PHB']) / 9.81
    gpi = (sel_rename(gp, 0, 2) + sel_rename(gp, 1, 3)) / 2
    return Dataset({'T': T, 'GP': gpi})
