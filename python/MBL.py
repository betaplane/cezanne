import xarray as xr
import numpy as np
import wrf


ds = xr.open_dataset('/nfs/temp/sata1_modelosglobales/WRFOUT_OPERACIONAL/c01_2018031012/wrfout_d03_2018-03-10_12:00:00')

# this computes temp from pot temp
# 300. is base state pot temp, 'PB' is base state pressure
T = wrf.tk(ds['P'] + ds['PB'], ds['T'] + 300.)
gp = (ds['PH'] + ds['PHB']) / 9.81

LM = ds['LANDMASK'][0,:,:]
t = T.where(gp.values[:,1:,:,:]<3000, 999).where(LM==0, 999)
i = t.argmin('bottom_top')
tm = t.sel(bottom_top=i)

j = np.broadcast_to(np.arange(t.bottom_top.shape[0]).reshape((1, -1, 1, 1)), t.shape)
j = xr.DataArray(j, coords=t.coords)

mbl = j.where(t-tm >= 0.5)
mbl = mbl.where(mbl > i).min('bottom_top')

dt = t.diff('bottom_top')
i = j[:,1:,:,:].where(dt > 0).min('bottom_top')         # layer index in which temp gradient is first positive
MBL = gp.isel(bottom_top_stag=i.fillna(0)).where(LM==0) # bottom interface of that layer
