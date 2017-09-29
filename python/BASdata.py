#!/usr/bin/env python
import requests, re, os
from bs4 import BeautifulSoup
import pandas as pd

sfc = "https://legacy.bas.ac.uk/met/READER/surface/stationpt.html"
sfc_wind = "https://legacy.bas.ac.uk/met/READER/surface/stationwind.html"
aws = "https://legacy.bas.ac.uk/met/READER/aws/awspt.html"
aws_wind = "https://legacy.bas.ac.uk/met/READER/aws/awswind.html"

def data(url):
    r = requests.get(url)
    s = BeautifulSoup(r.text, 'html5lib')
    dname = os.path.dirname(url)

    data = []
    for l in s.find_all(href=re.compile('All|\d\d')):
        h = l.attrs['href'].split('.')
        a = requests.get(os.path.join(dname, l.attrs['href']))
        txt = BeautifulSoup(a.text, 'html5lib').find(href=re.compile('txt'))
        try:
            url = os.path.join(dname, txt.attrs['href'])
            d = pd.read_csv(url, index_col=0, header=None, skiprows=1, na_values='-', delim_whitespace=True).stack().to_frame()
            d.index = pd.DatetimeIndex(['{}-{}'.format(y, m) for y, m in d.index.tolist()])
            d.columns = pd.MultiIndex.from_tuples([h[:3]])
        except:
            print('error:', h[:3])
        else:
            print(h[:3])
            data.append(d)
    return data

# data = pd.concat([parse(l) for l in s.find_all(href=re.compile('All|\d\d'))], 1)

def stations(url):
    r = requests.get(url)
    s = BeautifulSoup(r.text, 'html5lib')

    for body in s('tbody'):
        body.unwrap()

    t = pd.read_html(str(s.find('table')), index_col=0, header=0)[0][['Name', 'Latitude', 'Longitude', 'Height']]
    t.index.name = 'id'

    def parse_num(x):
        num, ext = float(x[:-1]), x[-1]
        return -num if (ext == 'S') or (ext == 'W') else num

    for c in set(t.columns) - {'Name'}:
        t[c] = t[c].apply(parse_num)

    return t


# with pd.HDFStore('../../data/BAS/READER.h5') as store:
#     store['sfc'] = data
#     store['sta'] = t
