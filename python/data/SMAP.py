"""
SMAP soil moisture data
-----------------------
"""
import os, requests, re, os, tables
from io import BytesIO
from datetime import datetime
from traitlets.config import Application
from traitlets.config.loader import PyFileConfigLoader, ConfigFileNotFound
from traitlets import Unicode, Instance, Float, Integer
from bs4 import BeautifulSoup
from glob import glob
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm


class SessionWithHeaderRedirection(requests.Session):
    """Requests subclass which maintains headers when redirected (see https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python), to access EarthData servers)

    """
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

class EarthData(Application):
    """Class to hold EarthData credentials and create a :class:`SessionWithHeaderRedirection`.

    """
    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    username = Unicode().tag(config=True)
    "EarthData username"
    password = Unicode().tag(config=True)
    "EarthData password"
    url = Unicode().tag(config=True)
    "url of the particular dataset to be downloaded"

    def __init__(self, config={}, **kwargs):
        cfg = PyFileConfigLoader(os.path.expanduser(self.config_file)).load_config()
        cfg.merge(config)
        super().__init__(config=cfg, **kwargs)
        self.session = SessionWithHeaderRedirection(self.username, self.password)

    def __del__(self):
        self.session.close()


class SMAP(EarthData):
    """EarthData download app (right now, specifically SMAP data). Main method is :meth:`.get`. See :class:`EarthData` for possible init arguments in addition to the :mod:`traitlets <traitlets.config>` class attributes.

    """
    from_date = Instance(datetime).tag(config=True)
    "earlist date for which to download data"
    to_date = Instance(datetime, datetime.now().timetuple()[:3]).tag(config=True)
    "latest date for which to download data (default: now)"
    lon_name = Unicode().tag(config=True)
    "name of the longitude variable in the downloaded HDF5 dataset"
    lat_name = Unicode().tag(config=True)
    "name of the latitude variable in the downloaded HDF5 dataset"
    var_name = Unicode().tag(config=True)
    "name of the data variable in the downloaded HDF5 dataset"
    group = Unicode().tag(config=True)
    "basename (without ending in '_AM'/'_PM') of the HDF5 group containing the data and coordinate variables"
    missing_value = Float().tag(config=True)
    "missing value flag"
    lon_extent = Instance(slice).tag(config=True)
    "slice of longitudes to :meth:`xarray.DataArray.sel`"
    lat_extent = Instance(slice).tag(config=True)
    "slice of latitudes to :meth:`xarray.DataArray.sel`"
    write_interval = Integer(100).tag(config=True)
    "number of days between write-outs of the downloaded data"
    outfile = Unicode().tag(config=True)
    "output base file name (without .nc extension)"

    def listdir(self, url):
        r = self.session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        return set([n.get('href') for n in soup.find_all('a')])

    def get(self):
        """Get SMAP data from the url specified in the :attr:`EarthData.config_file` or as the :mod:`configurable <traitlets.config>` :attr:`EarthData.url` traitlet. Retrieved data is appended to an :class:`xarray.DataArray` accessible as :attr:`.x` on this class.

        """
        def f(link):
            try:
                m = re.match('\d{4}\.\d\d\.\d\d', link).group(0)
                d = datetime.strptime(m, '%Y.%m.%d')
                assert d >= self.from_date
                assert d <= self.to_date
            except:
                return False
            return (link, d)

        links = [f(l) for l in self.listdir(self.url)]
        links = [l for l in links if l]

        self.fileno = 0
        g = glob('{}*'.format(self.outfile))
        if len(g) > 0:
            self.fileno = max([int(re.search('_(\d+)\.nc', h).group(1)) for h in g]) + 1
            with xr.open_mfdataset(g) as ds:
                links = [l for l in links if l[1].date() not in np.unique(ds.indexes['time'].date)]

        for i, (l, d) in enumerate(tqdm(links)):
            h5 = [h for h in self.listdir(os.path.join(self.url, l)) if h[-2:] == 'h5'][0]
            r = self.session.get(os.path.join(self.url, l, h5), stream=True)
            r.raise_for_status()
            self.append_data(r, d)
            if (i+1) % self.write_interval == 0:
                self.x.to_dataset(name=self.var_name).to_netcdf('{}_{}.nc'.format(self.outfile, self.fileno))
                self.fileno += 1
                del self.x
        if hasattr(self, 'x'):
            self.x.to_dataset(self.var_name).to_netcdf('{}_{}.nc'.format(self.outfile, self.fileno))

    def coords(self, coord, axis):
        c = np.ma.masked_equal(coord, self.missing_value)
        x = c.take(0, axis=axis)
        for i in range(1, c.shape[axis]):
            x = np.ma.masked_equal(np.where(x.mask, c.take(i, axis), x), self.missing_value)
        assert not x.mask, "missing coordinates"
        return x.data

    def to_xarray(self, h5, am_pm, date):
        group = '{}_{}'.format(self.group, am_pm)
        ext = '_pm' if am_pm == 'PM' else ''
        lon = self.coords(h5.get_node(os.path.join(group, '{}{}'.format(self.lon_name, ext))), 0)
        lat = self.coords(h5.get_node(os.path.join(group, '{}{}'.format(self.lat_name, ext))), 1)
        x = np.ma.masked_equal(h5.get_node(
            os.path.join(group, '{}{}'.format(self.var_name, ext))), self.missing_value)
        X = xr.DataArray(x.filled(np.nan), coords=[('lat', lat.flatten()), ('lon', lon.flatten())])
        X = X.sel(lon=self.lon_extent, lat=self.lat_extent).expand_dims('time')
        X['time'] = ('time', pd.DatetimeIndex([date]) + pd.Timedelta({'AM': 0, 'PM': '12h'}[am_pm]))
        return X

    def append_data(self, resp, date):
        b = BytesIO()
        for chunk in resp.iter_content(chunk_size=1024**2):
            b.write(chunk)
        b.seek(0)
        h5 = tables.open_file('unimportant.h5', driver='H5FD_CORE', driver_core_image=b.read(), driver_core_backing_store=0)
        for am_pm in ['AM', 'PM']:
            x = self.to_xarray(h5, am_pm, date)
            try:
                self.x = xr.concat((self.x, x), 'time')
            except AttributeError:
                self.x = x

if __name__ == '__main__':
    pass