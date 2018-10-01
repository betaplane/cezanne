#!/usr/bin/env python
import os
from traitlets.config import Application, Config
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode, Integer
from importlib import import_module
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from warnings import catch_warnings, simplefilter
# import sys
# sys.path.insert(0, '/usr/local/lib/python3.6/site-packages/GDAL-2.1.3-py3.6-macosx-10.12-x86_64.egg/')
# import gdal

# for image textures:
# http://geoexamples.blogspot.com/2014/02/3d-terrain-visualization-with-python.html

class Surface(Application):
    config_file = Unicode('./config.py').tag(config=True)
    wrf_file = Unicode().tag(config=True)

    def __init__(self, config={}, **kwargs):
        try:
            cfg = PyFileConfigLoader(os.path.abspath(self.config_file)).load_config()
            cfg.merge(config)
        except ConfigFileNotFound:
            cfg = Config(config)
        super().__init__(config=cfg, **kwargs)

    def initialize(self):
        self.nc = Dataset(self.wrf_file)
        X, Y = self.nc['XLONG'][0, : ,:], self.nc['XLAT'][0, :, :]
        self.affine = self.create_affine(X, Y)
        self.xbnds = X.min(), X.max()
        self.ybnds = Y.max(), Y.min()

    def __del__(self):
        try:
            self.nc.close()
        except: pass

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
        gdal = import_module('gdal')
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
        y = y + self.nc.dimensions['south_north'].size
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


class Visualization(Surface):
    dem_file = Unicode('dem_d03_12.nc').tag(config=True)
    "name of DEM file (in netcdf fornat, as produced by :meth:`dem`)"

    image_file = Unicode('marble_d03_rot.jpg').tag(config=True)
    "name of image (extension .jpg or .png) to use as texture over the DEM"

    var_name = Unicode('CLDFRA').tag(config=True)
    "name of variable to use for volume rendering"

    movie_file = Unicode('out.mp4').tag(config=True)
    "name of the movie file to be written out (extension .mp4 or .gif)"

    fps = Integer(1).tag(config=True)
    "frame rate of the movie file to be written out"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self, offscreen=False):
        super().initialize()
        tvtk = import_module('tvtk.api')
        self.mlab = import_module('mayavi.mlab')
        self.mlab.options.offscreen = offscreen

        ds = xr.open_dataset(self.dem_file)

        # here I 'fill' the DEM westward with zeros (over the ocean)
        dx = ds.lon.diff('lon').mean().item()
        i = np.arange(ds.lon.min(), self.xbnds[0], -dx)[:0:-1]
        Z = xr.DataArray(np.pad(ds.z, [(0, 0), (len(i), 0)], 'constant'),
                         coords = [('lat', ds.lat), ('lon', np.r_[i, ds.lon])])
        self.z = self.project_xarray(Z)

        # get the texture from image (extensions either 'jpg' or 'png')
        reader = {'.jpg': tvtk.tvtk.JPEGReader, '.png': tvtk.tvtk.PNGReader}[os.path.splitext(self.image_file)[1]]
        im = reader(file_name = self.image_file)
        self.tex = tvtk.tvtk.Texture(input_connection = im.output_port, interpolate = 1)

        self.fig = self.mlab.figure(size=(800, 800))

        # surf = self.mlab.surf(self.z.x.T, self.z.y.T, self.z.z.values.T/1000, colormap='gist_earth')
        surf = self.mlab.surf(self.z.x.T, self.z.y.T, self.z.z.values.T/1000, color=(1, 1, 1))
        surf.actor.enable_texture = True
        surf.actor.tcoord_generator_mode = 'plane'
        surf.actor.actor.texture = self.tex

        # self.mlab.view(-110, 80, 115, [50, 60, 1])
        self.mlab.view(-120, 65, 200, [40, 66, -6])

        ly, lx, self.nt = [self.nc.dimensions[n].size for n in ['south_north', 'west_east', 'Time']]
        self.xyz = [c.transpose(2, 1, 0) for c in np.mgrid[0:10:0.05, :ly, :lx]][::-1]
        self.vol = None

        @self.mlab.animate
        def anim():
            for i in range(self.nt):
                self.anim_func(i)
                yield

        self.anim = anim

    def anim_func(self, i):
        wrf = import_module('wrf')
        iz = np.arange(0, 10, .05)
        if self.vol is not None:
            self.vol.remove()
        cld = wrf.vinterp(self.nc, wrf.getvar(self.nc, self.var_name, timeidx=int(i)), 'ght_msl', iz)
        xyzc = self.xyz + [cld.values.transpose(2, 1, 0)]
        self.vol = self.mlab.pipeline.volume(self.mlab.pipeline.scalar_field(*xyzc), color=(1, 1, 1), figure=self.fig)
        return self.mlab.screenshot(antialiased=True)

    def write_movie(self):
        mp = import_module('moviepy.editor')
        vc = mp.VideoClip(self.anim_func, duration=self.nt / self.fps)
        writer = {'.mp4': 'write_videofile', '.gif': 'write_gif'}[os.path.splitext(self.movie_file)[1]]
        getattr(vc, writer)(self.movie_file, fps=self.fps)

    def start(self):
        # http://docs.enthought.com/mayavi/mayavi/auto/example_offscreen.html
        from mayavi.api import OffScreenEngine
        from mayavi.tools.sources import scalar_field
        from mayavi.modules.api import Volume
        eng = OffScreenEngine()
        scene = eng.new_scene()
        eng.add_source(scalar_field())

if __name__ == '__main__':
    app = Visualization(surface=True)
    app.parse_command_line()
    app.initialize(offscreen=True)
    app.anim_func(0)
    app.mlab.savefig('test.png')
