#!/usr/bin/env python
import pandas as pd
import numpy as np
from io import BytesIO
from pycurl import Curl
import re


# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
# https://climatedataguide.ucar.edu/climate-data/overview-climate-indices


urls = {
    'aao_mly': 'http://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii',
    'aao_dly': 'ftp://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.aao.index.b790101.current.ascii',
    'sam_mly': 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt',
    'sam_sea': 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.seas.txt',
    'nino34': 'https://www.esrl.noaa.gov/psd/data/correlation/nina34.data',
    'nino34_1870': 'https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/nino34.long.data',
    'nino4': 'https://www.esrl.noaa.gov/psd/data/correlation/nina4.data',
    'nino12': 'https://www.esrl.noaa.gov/psd/data/correlation/nina1.data',
    'nino_oni': 'https://www.esrl.noaa.gov/psd/data/correlation/oni.data',
    'nino_tni': 'https://www.esrl.noaa.gov/psd/data/correlation/tni.data',
    'soi': 'http://www.cpc.ncep.noaa.gov/data/indices/soi'
}


def getc(url):
    buf = BytesIO()
    c = Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buf)
    c.perform()
    c.close()
    return buf

def aao(key, buf):
    c, s = {'mly': [2, '{:4d}-{:02d}'], 'dly': [3, '{:4d}-{:02d}-{:02d}']}[key[-3:]]
    df = pd.read_csv(buf, delim_whitespace=True, header=None).dropna()
    df.index = pd.DatetimeIndex([s.format(*[int(k) for k in j]) for i, j in df.iloc[:,:c].iterrows()])
    return df.drop(range(c), 1)

def sam(key, buf):
    df = pd.read_csv(buf, delim_whitespace=True)
    if key[-3:] == 'sea':
        return df
    df = df.stack()
    df.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in df.index.tolist()])
    return df

def nino(key, buf):
    i = 0
    while True:
        l = buf.readline()
        if re.match(b' *-99.9', l):
            i = 0
        i += 1
        if l == b'':
            break
    skip = i - 1
    buf.seek(0)
    df = pd.read_csv(buf, 
                     delim_whitespace=True,
                     skiprows=1,
                     skipfooter=skip,
                     index_col=0,
                     na_values=-99.99,
                     header=None).stack()
    df.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in df.index.tolist()])
    return df

def soi(buf):
    with BytesIO() as bio:
        bio.writelines([l for l in buf if re.search(b'\d', l)])
        df = pd.read_fwf(bio, widths = [4] + 12 * [6], header = None, index_col = 0, na_values = -999.9)
        i = np.where(np.diff(df.index) != 1)[0][0] + 1
        df1 = df.iloc[:i].stack()
        df2 = df.iloc[i:].stack()
        df1.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in df1.index.tolist()])
        df2.index = pd.DatetimeIndex(['{}-{}'.format(*i) for i in df2.index.tolist()])

def download(name):
    with pd.HDFStore(name) as f:
        for k, url in urls.items():
            print('downloading {}'.format(k))
            buf = getc(url)
            buf.seek(0)
            if k == 'soi':
                f['soi'], f['soi_stand'] = soi(buf)
            else:
                f[k] = {'aao': aao, 'sam': sam, 'nin': nino}[k[:3]](k, buf)
            buf.close()
