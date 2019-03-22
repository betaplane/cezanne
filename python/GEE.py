"""
Google Earth Engine
===================

A module that demonstrates how to use the GEE python API and implements some wrapping.

"""
import ee, os
from ee import mapclient
import numpy as np
from geo import Loc
from traitlets import List, Unicode, Integer
from importlib import import_module
from cartopy.io.shapereader import Reader
from cartopy import crs

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

    def __init__(self, collection=None, copy=None, sort=True):
        if copy is not None:
            for k, v in copy.__dict__.items():
                setattr(self, k, v)
        else:
            ee.Initialize() # as long as we use the 'copy' hack - later could be just module leve call
            self.ic = ee.ImageCollection(collection)
            if sort:
                self.ic = self.ic.sort('system:time_start')
            self._list = iList(self.ic)

    def __getitem__(self, n):
        return self._list[n]

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

    def list_properties(self, prop_name, collection=None):
        c = self.ic if collection is None else collection
        return c.iterate(lambda i, l: ee.List(l).add(i.get(prop_name)), []).getInfo()

    def time_series(self, reducer=None, geometry=None, collection=None):
        c = self.ic if collection is None else collection
        g = self.shelf if geometry is None else geometry
        r = ee.Reducer.mean() if reducer is None else reducer
        return c.iterate(lambda i, l: ee.List(l).add(i.reduceRegion(reducer=r, geometry=g)), []).getInfo()

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
        def f(im):
            im.visualize(**vis_params)
        coll = self.ic.map(lambda i: i.visualize(**vis_params))
        job = ee.batch.Export.movie.toDrive(coll, description=name, region=self.bbox_ring(), framesPerSecond=fps)
        job.start()
        return job


class Mueller(Loc, GEE):

    path = Unicode('').tag(config=True)
    map_center = List([-66.8, -67.3, 10]).tag(config=True)
    coastline_file = Unicode('').tag(config=True)
    shelf_features = List([]).tag(config=True)
    quanta_epsg = Integer().tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'copy' not in kwargs:
            self.set_bbox()

    def set_bbox(self):
        self.ic = self.ic.filterBounds(ee.Geometry.Rectangle(self.bbox))
        self.meta()

    @property
    def shelf(self):
        """returns Mueller ice shelf as :class:`shapely.geometry.Polygon` based on :attr:`coastline_file`"""
        R = Reader(os.path.join(self.path, self.coastline_file))
        recs = [r for r in R.records() if r.attributes['FID_Coastl'] in self.shelf_features]
        # order is important
        geoms = [[r.geometry for r in recs if r.attributes['FID_Coastl']==f][0] for f in self.shelf_features]
        pts = np.hstack([np.r_['1,2', p.xy] for g in geoms for p in g.geoms]).T.tolist()
        # important! planar geometry ('False')
        # https://developers.google.com/earth-engine/geometries_planar_geodesic
        return ee.Geometry.Polygon(pts, 'EPSG:{}'.format(self.quanta_epsg), False)


def write_times_on_movie(filename, times, strings):
    cv2 = import_module('cv2')
    tqdm = import_module('tqdm')
    cap = cv2.VideoCapture(filename)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        os.path.join(os.path.split(filename)[0], 'out.mp4'),
        int(cap.get(cv2.CAP_PROP_FOURCC)),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (w, h)
    )

    frames = []
    with tqdm.tqdm(total = 2*n) as prog:
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            prog.update(1)

        for i in np.argsort(times):
            frame = frames[i]
            s = str(times[i]) + '  ' + strings[i]
            cv2.putText(frame, s, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2,cv2.LINE_AA)
            out.write(frame)
            prog.update(1)

    out.release()
    cap.release()
