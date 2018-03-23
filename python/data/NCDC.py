"""
NCDC webservice
---------------
"""
#!/usr/bin/env python
import requests, os, json, re, tarfile
from bs4 import BeautifulSoup
from io import BytesIO
import pandas as pd
import numpy as np
import xarray as xr


def purge_extracted(dir, retain):
    for f in os.listdir(dir):
        if os.path.splitext(f)[0] not in retain:
            os.remove(os.path.join(dir, f))



base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
#https://www.ncdc.noaa.gov/cdo-web/webservices/ncdcwebservices

def parse_stations(record):
    rec = record.copy()
    full_id = rec.pop('id')
    num_id = re.search('(\d+)', full_id.split(':')[1]).group(1)
    cols, data = zip(*rec.items())
    row = pd.Series(data, index=cols, name=int(num_id)).to_frame().T
    row['id'] = full_id
    return row

def parse_categories(idx, rec, categories):
    s = pd.Series({v['id']: True for v in rec}, name=idx)
    categories.update({v['id']: v['name'] for v in rec})
    return s


def get(endpoint, parser=None, **params):
    with requests.Session() as s:
        s.headers.update({'token': 'OpqrLypwpgTRWdUZmjVkZKIFNsfovPSx'})
        s.params.update({'limit': 50})

        params.update({'offset': 1})
        tot = 1
        results = []
        while params['offset'] <= tot:
            r = s.get(os.path.join(base_url, endpoint), params=params)
            if not r.ok:
                print(r.text)
                break
            try:
                j = json.loads(r.text)
                tot = j['metadata']['resultset']['count']
                params.update({'offset': params['offset'] + 50})
                results.extend(j['results'])
            except:
                break
    if parser is None:
        return results
    else:
        return pd.concat([parser(r) for r in results], 0)

class GHCND(object):
    """
Class to read in NCEI (National Centers for Environmental Information, formerly National Climatic Data Center, NCDC) `Global Historical Climatology Network - Daily (GHCND) <ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/>`_ data from tar archive containing all station data.
    """
    ghcnd_widths = np.r_[[11, 4, 2, 4], np.array([5, 1, 1, 1], ndmin=2).repeat(31, 0).flatten()]
    station_cols = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)]

    def __init__(self, filename, copy=None):
        if copy is None:
            self.tf = tarfile.open(filename)
            self.members = self.tf.getmembers()
        else:
            self.tf = copy.tf
            self.members = copy.members

    def find(self, s):
        info = [m for m in self.members if re.search(s, m.name)]
        if len(info)==1:
            return info[0]
        raise Exception('Multiple files found.')

    def data(self, info):
        info = info if isinstance(info, tarfile.TarInfo) else self.find(str(info))
        bio = BytesIO(self.tf.extractfile(info).read())
        return self.convert(bio)

    @classmethod
    def read_dir(cls, dir):
        l = os.listdir(dir)
        idx = pd.Index([os.path.splitext(f)[0] for f in l], name='station')
        return xr.concat([cls.convert(os.path.join(dir, f)) for f in l], idx)

    @classmethod
    def convert(cls, bytesIO):
        df = pd.read_fwf(bytesIO, header=None, widths=cls.ghcnd_widths, na_values='-9999')
        rows = pd.MultiIndex.from_arrays((['{}-{}'.format(*d) for i, d in df[[1, 2]].iterrows()], df[3]),
                                    names=['months', 'field'])
        data = xr.concat([xr.DataArray(df.iloc[:, i::4], coords=[('rows', rows), ('days', range(1, 32))])
                          for i in range(4, 8)], pd.Index(['value', 'mflag', 'qflag', 'sflag'], name='flag'))
        data = data.unstack('rows').stack(time=('months', 'days'))
        data = data.sel(time=data.sel(flag='value').notnull().any('field').values)
        data['time'] = ('time', pd.DatetimeIndex(['{}-{}'.format(*d) for d in data.time.values]))
        return data.isel(time = data.time.argsort().values)

    @classmethod
    def read_stations(cls, filename):
        sta = pd.read_fwf(filename, header=None, colspecs=cls.station_cols)
        sta.columns = ['id', 'lat', 'lon', 'elev', 'state', 'name', 'GSN flag', 'HCN/CRN flag', 'WMO id']
        return sta


    def __del__(self):
        self.tf.close()


class GHCND2(object):
    ghcnd_widths = np.r_[[11, 4, 2, 4], np.array([5, 1, 1, 1], ndmin=2).repeat(31, 0).flatten()]
    station_cols = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)]

    def __init__(self, filename, copy=None):
        self.tf = tarfile.open(filename, 'r|gz')

    def extract(self, stations):
        l = []
        index = []
        for f in self.tf:
            not_found = True
            for s in stations:
                if re.search(str(s), f.name):
                    not_found = False
                    break
            if not_found:
                continue
            bio = BytesIO(self.tf.extractfile(f).read())
            l.append(self.convert(bio))
            index.append(s)
        return xr.concat(l, pd.Index(index, name='station'))

    def convert(self, bytesIO):
        df = pd.read_fwf(bytesIO, header=None, widths=self.ghcnd_widths, na_values='-9999')
        rows = pd.MultiIndex.from_arrays((['{}-{}'.format(*d) for i, d in df[[1, 2]].iterrows()], df[3]),
                                    names=['months', 'field'])
        data = xr.concat([xr.DataArray(df.iloc[:, i::4], coords=[('rows', rows), ('days', range(1, 32))]) 
                          for i in range(4, 8)], pd.Index(['value', 'mflag', 'qflag', 'sflag'], name='flag'))
        data = data.unstack('rows').stack(time=('months', 'days'))
        data = data.sel(time=data.sel(flag='value').notnull().any('field'))
        data['time'] = ('time', pd.DatetimeIndex(['{}-{}'.format(*d) for d in data.time.values]))
        return data.isel(time = data.time.argsort())

    def __del__(self):
        self.tf.close()

if __name__ == '__main__':
    r = get('stations', parser=parse_stations, extent='-90,-180,-50,180')

    with pd.HDFStore('../../data/Antarctica/READER.h5') as store:
        l = r.join(store['sta'], how='outer')

    cats = {}
    d = pd.concat(
        [parse_categories(i, get('datacategories', stationid=j.id), cats) for i, j in r.iterrows],
        1).T.fillna(False)

    l = l.merge(d, left_index=True, right_index=True, how='outer')
