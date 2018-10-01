import numpy as np
from importlib import import_module
import xarray as xr
import wrf
from netCDF4 import Dataset
from mayavi import mlab
from tvtk.api import tvtk
import sys
sys.path.insert(0, '/usr/local/lib/python3.6/site-packages/GDAL-2.1.3-py3.6-macosx-10.12-x86_64.egg/')
import gdal

# for image textures:
# http://geoexamples.blogspot.com/2014/02/3d-terrain-visualization-with-python.html

class LandSurf(object):
    def __init__(self, nc_path):
        self.nc = Dataset(nc_path)
        X, Y = self.nc['XLONG'][0, : ,:], self.nc['XLAT'][0, :, :]
        self.affine = self.create_affine(X, Y)
        self.xbnds = X.min(), X.max()
        self.ybnds = Y.max(), Y.min()

    def __del__(self):
        self.nc.close()

    def proj(self):
        pyproj = import_module('pyproj')
        return pyproj.Proj(
            lon_0 = self.nc.CEN_LON,
            lat_0 = self.nc.CEN_LAT,
            lat_1 = self.nc.TRUELAT1,
            lat_2 = self.nc.TRUELAT2,
            proj = 'lcc'
        )

    def create_affine(self, x, y):
        proj = self.proj()
        x, y = proj(x, y)
        m, n = x.shape
        j, i = np.mgrid[:m, :n]

        C = np.linalg.lstsq(
            np.r_[[x.flatten()], [y.flatten()], np.ones((1, m * n))].T,
            np.r_[[i.flatten()], [j.flatten()], np.ones((1, m * n))].T)[0].T
        b = C[:2, 2]
        A = C[:2, :2]

        def to_grid(lon, lat):
            lon, lat = proj(lon, lat)
            return A.dot(np.vstack((lon.flatten(), lat.flatten()))) + np.r_[[b]].T

        return to_grid

    def dem(self, dem_path, decimate=1):
        ds = gdal.Open(dem_path)
        g = ds.GetGeoTransform()
        b = ds.GetRasterBand(1)
        z = b.ReadAsArray()

        dx, dy, x = g[1], g[5], z
        if decimate > 1:
            sig = import_module('scipy.signal')
            x = sig.decimate(sig.decimate(z, decimate, axis=0), decimate, axis=1)
            dx = (g[1] * z.shape[1]) / x.shape[1]
            dy = (g[5] * z.shape[0]) / x.shape[0]

        Z = xr.DataArray(x, coords=[
            ('lat', g[3] + dy * (np.arange(x.shape[0]) + .5)),
            ('lon', g[0] + dx * (np.arange(x.shape[1]) + .5))
        ]).sel(lon=slice(*self.xbnds), lat=slice(*self.ybnds))

        return self.project_xarray(Z)

    def project_xarray(self, Z):
        x, y = self.affine(*np.meshgrid(Z.lon, Z.lat))
        return xr.Dataset({
            'z': Z,
            'x': (('lat', 'lon'), x.reshape(Z.shape)),
            'y': (('lat', 'lon'), y.reshape(Z.shape))
        })

    def image(self, im_path):
        Image = import_module('PIL.Image')
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(im_path)
        # blue marble next gen is 1/4th of an arc minute (1/dx = 240)
        i = (np.vstack((self.xbnds, self.ybnds)).T * [1, -1] + [90, 0]) * 240
        return im.crop(np.hstack((np.floor(i[0, :]), np.ceil(i[1, :]))))


class Vis(LandSurf):
    def __init__(self, nc_path, dem_path):
        super().__init__(nc_path)
        ds = xr.open_dataset(dem_path)
        dx = ds.lon.diff('lon').mean().item()
        i = np.arange(ds.lon.min(), self.xbnds[0], -dx)[:0:-1]
        Z = xr.DataArray(np.pad(ds.z, [(0, 0), (len(i), 0)], 'constant'),
                         coords = [('lat', ds.lat), ('lon', np.r_[i, ds.lon])])
        self.z = self.project_xarray(Z)

    def show(self, var_name, im_path):
        fig = mlab.figure(size=(800, 800))
        # png = tvtk.PNGReader(file_name = im_path)
        # tex = tvtk.Texture(input_connection = png.output_port, interpolate = 1)

        # surf = mlab.surf(self.z.x, self.z.y, self.z.z.values/5e4, color=(1, 1, 1))
        # surf.actor.enable_texture = True
        # surf.actor.tcoord_generator_mode = 'plane'
        # surf.actor.actor.texture = tex

        wrf = import_module('wrf')
        z = np.arange(0, 10, .05)
        cld = wrf.vinterp(self.nc, wrf.getvar(self.nc, var_name, timeidx=100), 'ght_msl', z)

        z, y, x = np.mgrid[0:10:0.05, :126, :78]
        tr = lambda x: x.transpose(1, 2, 0)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(tr(x), tr(y), tr(z)/50, tr(cld.values)), color=(1, 1, 1))

# mlab.view(0, 70, 200, [65, 35, 2])
