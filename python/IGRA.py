#!/usr/bin/env python
import pandas as pd
import numpy as np
import xarray as xr
import sys, re
from subprocess import Popen, PIPE
from pycurl import Curl
from io import BytesIO, StringIO
from zipfile import ZipFile
import tarfile

# raw observations
# curl -O https://www1.ncdc.noaa.gov/pub/data/igra/data/data-por/CIM00085586-data.txt.zip
# curl -O https://www1.ncdc.noaa.gov/pub/data/igra/data/data-por/ARM00087418-data.txt.zip

head = (
    ('HEADREC', 1, 1),
    ('ID', 2, 12),
    ('YEAR', 14, 17),
    ('MONTH', 19, 20),
    ('DAY', 22, 23),
    ('HOUR', 25, 26),
    ('RELTIME', 28, 31),
    ('NUMLEV', 33, 36),
    ('P_SRC', 38, 45),
    ('NP_SRC', 47, 54),
    ('LAT', 56, 62),
    ('LON', 64, 71)
)

cols = (
    ('LVLTYP1', 1, 1),
    ('LVLTYP2', 2, 2),
    ('ETIME', 4, 8),
    ('PRESS', 10, 15),
    ('PFLAG', 16, 16),
    ('GPH', 17, 21),
    ('ZFLAG', 22, 22),
    ('TEMP', 23, 27),
    ('TFLAG', 28, 28),
    ('RH', 29, 33),
    ('DPDP', 35, 39),
    ('WDIR', 41, 45),
    ('WSPD', 47, 51)
)


def uz(x):
    c,a,b = zip(*x)
    d = list(zip(np.array(a)-1,b))
    return d,c

def eat(head, data, var):
    i = [np.repeat(pd.datetime(*t[1:]),t[0]) for t in
         zip(head['NUMLEV'], head['YEAR'], head['MONTH'], head['DAY'], head['HOUR'])]
    t = pd.DatetimeIndex([j for k in i for j in k])
    data.index = pd.MultiIndex.from_arrays([t,data['PRESS']], names=('datetime', 'p'))
    data = data.drop(-9999, 0, 'p').drop('PRESS', 1)
    return data if var is None else data[var].unstack()

def extract(file, var=None):
    z = zipfile.ZipFile(file)
    i = z.infolist()[0]
    b = z.open(i).read()
    z.close()
    d,c = uz(head)
    p = Popen(['grep', '#'], stdin=PIPE, stdout=PIPE)
    out,err = p.communicate(input=b)
    with BytesIO(out) as g:
        H = pd.read_fwf(g,d,names=c)

    d, c = uz(cols)
    p = Popen(['sed', '-e', 's/^#.*$//'], stdin=PIPE, stdout=PIPE)
    out, err = p.communicate(input=b)
    with BytesIO(out) as g:
        D = pd.read_fwf(g, d, names=c).dropna(0,'all')

    return eat(H, D, var)

def get(name):
    base = 'https://www1.ncdc.noaa.gov/pub/data/igra/data/data-por/{}-data.txt.zip'
    buf = BytesIO()
    c = Curl()
    c.setopt(c.URL, base.format(name))
    c.setopt(c.WRITEDATA, buf)
    c.perform()
    c.close()
    z = ZipFile(buf)
    out = z.open(z.infolist()[0]).read()
    z.close()
    return out.decode()

def parse(string, var=None):
    l = string.splitlines()
    d, c = uz(head)
    H = pd.read_fwf(StringIO('\n'.join([r for r in l if r[0]=='#'])), d, names=c)
    d, c = uz(cols)
    D = pd.read_fwf(StringIO('\n'.join([r for r in l if r[0]!='#'])), d, names=c)
    return eat(H, D, var)

class Monthly(object):
    """
Some methods to parse monthly `IGRA (Integrated Global Radiosonde Archive) <https://www1.ncdc.noaa.gov/pub/data/igra/>`_ files (ending in '-mly.txt') collected into a class.
    """
    num = re.compile('(\d+)')
    name = re.compile('([^_]+)_(\d+)')
    station_cols = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 76), (77, 81), (82, 88)]

    @staticmethod
    def _pivot(groupby):
        df = groupby.copy()
        df.index = pd.DatetimeIndex(['{}-{}'.format(*d) for d in df[[1, 2]].as_matrix()], name='time')
        df = df.drop([0, 1, 2], 1)
        df.columns = ['lvl', 'value', 'count']
        return df.pivot(columns='lvl')

    @classmethod
    def txt_to_xarray(cls, filename, stations, numeric=True, zip=None):
        """Return xarray.DataArray from IGRA monthly data files. See also :meth:`tar_to_xarray`.

        :param filename: name of regular text file or zipped file (need to give *zip* argument too)
        :param stations: list of station ids (full string) to extract
        :param numeric: if True, set the station dimension of the returned xarray.DataArray to integer values (if False, retain the alphanumeric station code
        :param zip: if *filename* refers to a ZipFile, give here the name of the file to extract
        :rtype: :class:`xarray.DataArray`

        """
        if zip is not None:
            filename = BytesIO(ZipFile(filename).open(zip).read())
        df = pd.read_csv(filename, header=None, delim_whitespace=True)
        df = df[df[0].isin(stations)]
        da = xr.DataArray(df.groupby(0).apply(cls._pivot)).unstack('dim_0').unstack('dim_1')
        da = da.rename({'dim_0_level_0': 'station', 'dim_1_level_0': 'type'})
        if numeric:
            da['station'] = cls.station_to_int(da.station)
        return da

    @classmethod
    def _untar(cls, tarf, info, *args, **kwargs):
        var, hour = cls.name.search(info.name).groups()
        da = cls.txt_to_xarray(BytesIO(tarf.extractfile(info).read()), *args, **kwargs)
        da['time'] = da.time + pd.Timedelta(int(hour), 'h')
        da.name = var
        return da

    @classmethod
    def tar_to_xarray(cls, *args, tar='all', **kwargs):
        """Wrapper around :meth:`txt_to_xarray` for tarfiles, with same arguments plus *tar*.

        :param tar: If 'all', extract all files in archive, otherwise this is the name of the file to extract.
        :rtype: :class:`xarray.DataArray`

        """
        with tarfile.open(args[0]) as tarf:
            if tar == 'all':
                return xr.merge([cls._untar(tarf, m, *args[1:], **kwargs) for m in tarf.getmembers()])
            else:
                return cls.txt_to_xarray(BytesIO(tarf.extractfile(tar).read()), *args[1:], **kwargs)

    @classmethod
    def read_stations(cls, filename, **kwargs):
        """Parse `station-list <https://www1.ncdc.noaa.gov/pub/data/igra/https://www1.ncdc.noaa.gov/pub/data/igra/igra2-station-list.txt>`_ file from `IGRA <https://www1.ncdc.noaa.gov/pub/data/igra/https://www1.ncdc.noaa.gov/pub/data/igra/>`_.

        :param filename: file to parse.
        :rtype: :class:`pandas.DataFrame`

        """
        sta = pd.read_fwf(filename, header=None, colspecs=cls.station_cols)
        sta.columns = ['id', 'lat', 'lon', 'elev', 'state', 'name', 'first_year', 'last_year', 'nobs']
        sta.index = cls.station_to_int(sta.id)
        return sta

    @classmethod
    def station_to_int(cls, c):
        return [int(cls.num.search(a).group(1)) for a in c.values]

if __name__ == "__main__":
    df = extract(sys.argv[1])
    S = pd.HDFStore('IGRAraw.h5')
    S[sys.argv[2]] = df
    S.close()
