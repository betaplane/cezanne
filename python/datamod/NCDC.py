#!/usr/bin/env python
import requests, os, json, re, tarfile
from bs4 import BeautifulSoup
from io import BytesIO
import pandas as pd
import numpy as np
import xarray as xr

base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'


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

class NCDC(object):
    widths = np.r_[[11, 4, 2, 4], np.array([5, 1, 1, 1], ndmin=2).repeat(31, 0).flatten()]

    def __init__(self, filename):
        self.tf = tarfile.open(filename)
        self.members = self.tf.getmembers()

    def find(self, s):
        return [m for m in self.members if re.search(s, m.name)]

    def data(self, info):
        bio = BytesIO(self.tf.extractfile(info).read())
        df = pd.read_fwf(bio, header=None, widths=self.widths, na_values='-9999')
        rows = pd.MultiIndex.from_arrays((['{}-{}'.format(*d) for i, d in df[[1, 2]].iterrows()], df[3]),
                                    names=['months', 'field'])
        data = xr.concat([xr.DataArray(df.iloc[:, i::4], coords=[('rows', rows), ('days', range(1, 32))]) 
                          for i in range(4, 8)], pd.Index(['value', 'mflag', 'qflag', 'sflag'], name='flag'))
        data = data.unstack('rows').stack(time=('months', 'days'))
        data = data.sel(time=data.sel(flag='value').notnull().squeeze())
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
