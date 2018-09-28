import numpy as np
from netCDF4 import Dataset
import wrf
from mayavi import mlab
from pyproj import Proj

# for image textures:
# http://geoexamples.blogspot.com/2014/02/3d-terrain-visualization-with-python.html

def proj_params(file, proj='lcc'):
    return {
        'lon_0': file.CEN_LON,
        'lat_0': file.CEN_LAT,
        'lat_1': file.TRUELAT1,
        'lat_2': file.TRUELAT2,
        'proj': proj
    }

ds = Dataset('/nfs/HPC/Prueba1_WRF/sinSST/WRFOUT/20180824-12_1p00/wrfout_d03_2018-08-24_12:00:00')
# proj should output projected coordinates in m
proj = Proj(**proj_params(ds))
X, Y = proj(ds['XLONG'][0, : ,:], ds['XLAT'][0, :, :])
hgt = ds['HGT'][0, :, :]
Z = np.arange(0, 10040, 50) / 1000
cld = wrf.vinterp(ds, wrf.getvar(ds, 'QVAPOR', timeidx=100), 'ght_msl', Z)

z, x, y = np.broadcast_arrays(Z.reshape((-1, 1, 1)), X, Y)

tr = lambda x: x.transpose(1, 2, 0)

fig = mlab.figure(size=(800, 800))
# mlab.surf(X, Y, z, colormap='gist_earth')
mlab.pipeline.volume(mlab.pipeline.scalar_field(tr(x), tr(y), tr(z), tr(cld.values)))

mlab.view(0, 70, 12000, [-166711, -260000, 2000])
