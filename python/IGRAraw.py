#!/usr/bin/env python
import pandas as pd
import numpy as np
import zipfile, io, sys, re
from subprocess import Popen, PIPE
 
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


def extract(file, var=None):
    def uz(x):
        c,a,b = zip(*x)
        d = list(zip(np.array(a)-1,b))
        return d,c
    z = zipfile.ZipFile(file)
    i = z.infolist()[0]
    b = z.open(i).read()
    z.close()
    d,c = uz(head)
    p = Popen(['grep', '#'], stdin=PIPE, stdout=PIPE)
    out,err = p.communicate(input=b)
    with io.BytesIO(out) as g:
        H = pd.read_fwf(g,d,names=c)

    d,c = uz(cols)
    p = Popen(['sed', '-e', 's/^#.*$//'], stdin=PIPE, stdout=PIPE)
    out,err = p.communicate(input=b)
    with io.BytesIO(out) as g:
        D = pd.read_fwf(g,d,names=c).dropna(0,'all')

    i = [np.repeat(pd.datetime(*t[1:]),t[0]) for t in
         zip(H['NUMLEV'], H['YEAR'], H['MONTH'], H['DAY'], H['HOUR'])]
    t = pd.DatetimeIndex([j for k in i for j in k])
    D.index = pd.MultiIndex.from_arrays([t,D['PRESS']], names=('datetime', 'p'))
    D = D.drop(-9999, 0, 'p').drop('PRESS', 1)
    return D if var is None else D[var].unstack()



if __name__ == "__main__":
    df = extract(sys.argv[1])
    S = pd.HDFStore('IGRAraw.h5')
    S[sys.argv[2]] = df
    S.close()
