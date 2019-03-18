"""
Google Earth Engine
===================

A module that demonstrates how to use the GEE python API and implements some wrapping.

"""
import ee
from ee import mapclient
import numpy as np
from geo import Loc
from traitlets import List
from importlib import import_module


class iList(object):

    max = 5000

    class image(ee.Image):
        def __init__(self, *args, fmt, **kwargs):
            super().__init__(*args, **kwargs)
            self._timestamp = self.date().format(fmt).getInfo()

        def __repr__(self):
            s = super().__repr__().split('>')[0]
            return '{} time: {}>'.format(s, self._timestamp)

    def __init__(self, collection):
        self._l = collection.toList(self.max)

    def __getitem__(self, n):
        return self.image(self._l.get(n), fmt=GEE.time_fmt)

class GEE(object):

    time_fmt = "YYYY-MM-dd'T'HH:mm:ss"

    def __init__(self, collection=None, copy=None):
        if copy is not None:
            for k, v in copy.__dict__.items():
                setattr(self, k, v)
        else:
            ee.Initialize() # as long as we use the 'copy' hack - later could be just module leve call
            self.ic = ee.ImageCollection(collection)

    def meta(self):
        self.size = self.ic.size().getInfo()
        self.properties = self.ic.propertyNames().getInfo()
        self.imageProperties = self.ic.toList(1).get(0).getInfo()
        self.start_end_1()

    def start_end_1(self):
        # suggested by https://developers.google.com/earth-engine/ic_info
        # seems to work better when filtered data is used
        dlim = self.ic.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
        self.start = np.datetime64(ee.Date(dlim.get('min')).format(self.time_fmt).getInfo())
        self.end = np.datetime64(ee.Date(dlim.get('max')).format(self.time_fmt).getInfo())

    def start_end_2(self):
        # I found this to be more in line with the collection's web catalog
        # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
        def f(d, newlist):
            return ee.List(newlist).add(ee.Date(d).format(self.time_fmt))
        self.start, self.end = np.array(ee.List(self.ic.get('date_range')).iterate(f, []).getInfo(), dtype='datetime64')

    def times(self, collection=None):
        def f(im, newlist):
            d = ee.Image(im).date().format(self.time_fmt)
            return ee.List(newlist).add(d)
        c = self.ic if collection is None else collection
        return np.array(c.iterate(f, []).getInfo(), dtype='datetime64[s]')

    def date(self, s, max=1000):
        a = np.datetime64(s)
        b = a + np.timedelta64(1, 'D')
        while True:
            f = self.ic.filterDate(np.datetime_as_string(a), np.datetime_as_string(b))
            if f.size().getInfo() > 0:
                break
            a = a - np.timedelta64(1, 'D')
            b = b + np.timedelta64(1, 'D')
        c = iList(f.toList(max), self)
        print(c.times)
        return c


    def map(self, *args, **kwargs):
        mapclient.map_instance = mapclient.MapClient(*args, **kwargs)

    def addToMap(self, eeobj, center=False, **vis_params):
        """Example of `palettable <https://jiffyclub.github.io/palettable/>`_ palette use::

            from palettable.cartocolors.sequential import SunsetDark_6

            GEE.addToMap(image, {'bands': ['HH], 'min':-20, 'max':0, 'palette': SunsetDark_6.hex_colors})

        Note that palettes can only be used with single-band visualiztion.

        """
        try:
            assert(mapclient.map_instance.is_alive())
            center = True
        except:
            mapclient.map_instance = None
        if 'palette' not in vis_params and hasattr(self, 'palette'):
            vis_params['palette'] = self.palette.hex_colors
        mapclient.addToMap(eeobj, vis_params)
        mapclient.centerMap(*self.map_center)

    def save(self, filename):
        mapclient.map_instance.canvas.postscript(file=filename)

    @staticmethod
    def LineString(df):
        return ee.Feature(ee.Geometry(df[['lon', 'lat']].as_matrix().tolist()))

    def scatter(self, lon, lat, c, cmap, nbins=100, vmin=None, vmax=None, **vis_params):
        plt = import_module('matplotlib.pyplot')
        colors = import_module('matplotlib.colors')
        tqdm = import_module('tqdm')
        cm = plt.get_cmap(cmap, nbins)
        c = np.array(c)
        mi = c.min() if vmin is None else vmin
        ma = c.max() if vmax is None else vmax
        x = np.floor((c - mi) / (ma - mi) * nbins)
        x[x==nbins] = nbins - 1

        for i in tqdm.tqdm(range(nbins)):
            j = x==i
            coll = ee.FeatureCollection([
                ee.Feature(ee.Geometry.MultiPoint(list(zip(lon[j], lat[j]))))
            ])
            self.addToMap(coll.draw(color=colors.rgb2hex(cm(i))), **vis_params)

    def bbox_ring(self):
        a, b, c, d = self.bbox
        return [(a, c), (b, c), (b, d), (a, d)]

    def exportImage(self, image, filename):
        job = ee.batch.Export.image.toDrive(image, fileNamePrefix=filename, region=self.bbox_ring())
        job.start()
        return job

    def exportMovie(self, name, fps=6, **vis_params):
        coll = self.ic.map(lambda i: i.visualize(**vis_params))
        job = ee.batch.Export.movie.toDrive(coll, description=name, region=self.bbox_ring(), framesPerSecond=fps)
        job.start()
        return job

class Mueller(Loc, GEE):

    map_center = List([-66.8, -67.3, 10]).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'copy' not in kwargs:
            self.set_bbox()

    def set_bbox(self):
        self.ic = self.ic.filterBounds(ee.Geometry.Rectangle(self.bbox))
        self.meta()

