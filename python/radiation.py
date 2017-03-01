#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate as ip
# from scipy.stats import binned_statistic
from functools import partial
import ephem as ep
import helpers as hh
from mapping import basemap
from interpolation import interp4D


D = pd.HDFStore('../../data/tables/station_data_new.h5')
rs = D['rs_w'].xs('prom', level='aggr', axis=1)
sta = D['sta']
T = hh.extract(D['ta_c'], 'prom', C2K=True)


So = 1361
sun = ep.Sun()
rad = np.pi / 180

def stationize(df):
    try:
        return pd.DataFrame(df, columns=df.columns.get_level_values('station'))
    except:
        return pd.DataFrame(df, index=df.index.get_level_values('station')


# hour angle
def hour(obs, t):
    obs.date = t
    sun.compute(obs)
    return obs.sidereal_time() - sun.ra

# declination
def dec(obs, t):
    obs.date = t
    sun.compute(obs)
    return sun.dec


# extra-terrestrial radiation
# Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. FAO Irrigation and drainage paper No. 56.
# Rome: Food and Agriculture Organization of the United Nations 56, 97–156.
def extraterrestrial(stations, time):
    t = pd.date_range(time[0], time[-1] + np.timedelta64(1, 'h'), freq='H')
    Ra = pd.DataFrame(index=time)
    for c, r in sta.iterrows():
        print(c)
        obs = ep.Observer()
        obs.lon = r.lon * rad
        obs.lat = r.lat * rad
        obs.elevation = r.elev
        f = partial(hour, obs)
        g = partial(dec, obs)
        # this is to account for the averaging period in the database
        te = t + np.timedelta64(4, 'h') - np.timedelta64(int(r.interval), 's')
        w = np.array([f(i) for i in te])
        d = np.array([g(i) for i in te])
        ws = np.arccos(-np.tan(obs.lat) * np.tan(d))
        w[w > np.pi] -= 2 * np.pi
        w[w < -np.pi] += 2 * np.pi
        w[w < -ws] = 0
        w[w > ws] = 0
        w1 = w[:-1]
        w2 = w[1:]
        d = (d[:-1] + d[1:]) / 2
        R = 12 / np.pi * So * (1 + 0.033 * np.cos(2 * np.pi / 365 * t[:-1].dayofyear)) * \
        ((w2 - w1) * np.sin(obs.lat) * np.sin(d) + np.cos(obs.lat) * np.cos(d) * (np.sin(w2) - np.sin(w1)))
        R[R < 0] = 0
        Ra = pd.concat((Ra, pd.Series(R, index=rs.index, name=c)), axis=1)
    return Ra


def solalt(obs, t):
    """
    Returns sun's zenith angle based on PyEphem Observer and time, and as convenience also the
    earth-sun distance in AU
    See http://rhodesmill.org/pyephem/quick.html#bodies.
    """
    obs.date = t
    sun.compute(obs)
    return (sun.alt, sun.earth_distance)


def extraterrestrial2(stations, t):
    S0 = 1370
    sun = ep.Sun()
    rad = np.pi / 180
    def per_station(c,r):
        print(c)
        obs = ep.Observer()
        obs.lon = r.lon * rad
        obs.lat = r.lat * rad
        obs.elevation = r.elev
        z = partial(solalt, obs)
        # here, shift the time vector by each station's averaging period so it contains the interval endpoints
        # add 1/2 hour to use midpoints (i.e. 270 mins together with UTC shift)
        te = t + np.timedelta64(270, 'm') - np.timedelta64(int(r.interval), 's')
        alt,d = np.array([z(i) for i in te]).T
        # compute averages over mu (= cos zenith, = sin alt) analytically
        mu = np.sin(alt)
        mu[mu<0] = 0
        return S0 * (1/d)**2 * mu
    return pd.DataFrame(dict([(c, per_station(c,r)) for c,r in stations.iterrows()]), index=time)


Ra = extraterrestrial2(sta, rs.index)
# Ra = D['Ra_w']

def Rso


def regression(rm, Ra):
    """Regress clear sky radiation onto theoretical extraterrestrial radiation.
    Selects brightest days in data somewhat arbitrarily to do the regression.

    :param rm: measured solar radiation
    :param Ra: extraterrestrial radiation
    :returns: DataFrame with columns 'b' (regression coefficients),
        'rms' (root mean square of residuals) and 'n' (number of samples used for regression)
    :rtype: DataFrame

    """
    b = []
    for c, r in rm.iteritems():
        x = pd.concat((Ra[c[0]], r), axis=1)
        # eliminate measured values > than extraterrestrial radiation
        x.iloc[:, 1][x.iloc[:, 1] > x.iloc[:, 0]] = np.nan
        # use Eq. 35 from Allen et al. as approximation and select days with > 0.9 relative sunshine duration
        xd = x.groupby(x.index.date).sum()
        n = 2 * (xd.iloc[:, 1] - 0.25 * xd.iloc[:, 0]) / xd.iloc[:, 0]
        n = n.reindex(x.index, method='ffill')
        y = x[n > .9]

        try:
            for i in range(2):
                X = y.dropna().as_matrix()
                lsq = np.linalg.lstsq(X[:, :1], X[:, 1:])
                a = lsq[0][0][0]
                err = lsq[1][0]
                n = y.count().min()
                # for second round, select only hours where measured radiation > predicted from first round
                y = x[x.iloc[:, 1] > Ra[c[0]] * a]
        except:
            b.append((np.nan, np.nan, 0))
        else:
            b.append((a, (err/n)**.5, n))
    return pd.DataFrame(b, index=rm.columns, columns=['b', 'rms', 'n'])


b = regression(rs, Ra).sort_index()
# manual selection of sensors at stations which have 2 - for now take the ones with lower RMS
from CEAZAMet import mult
c = b.loc[mult(rs)]
drop_codes = ['28', 'ANDACMP10', 'CACHRSTM', 'COMBRS', 'INILLARS', 'MINRS', 'PCRS2', 'TLHRSTM']
b.drop(drop_codes,level='code',inplace=True)
rs.drop(drop_codes,level='code',inplace=True,axis=1)

# very high regression coefficients are also suspicious
b.sort_values('b', inplace=True)
drop_codes.extend(['INIA66', 'PAGN', 'LMBT', 'SALH', ''])

def detect_breaks(df):
    i = df.dropna().index
    return pd.Series(np.diff(i), index=i[1:]).sort_values()

def residuals(code):
    return pd.concat((b['b'][code][0] * Ra[code], rs[code]), axis=1).diff(1, axis=1).iloc[:,1]

rs['TASC'][:'2014-12-07 14']

def itpl():
    "Interpolate 4D (true) temp and (mass level) geopotential fields to stations."
    nc = Dataset('../../data/WRF/2d/d02_2014-09-10_transf.nc')
    T = nc.variables['temp'][:]
    GP = nc.variables['ghgt'][:]

    ma = basemap(nc)
    x, y = ma.xy()
    ij = ma(*hh.lonlat(sta))
    t = hh.get_time(nc)

    Ti = interp4D((x, y), T, ij, sta.index, t, method='linear')
    Gi = interp4D((x, y), GP, ij, sta.index, t, method='linear')


S = pd.HDFStore('../../data/tables/LinearLinear.h5')
R = pd.HDFStore('../../data/tables/4Dfields.h5')
Z = S['z']['d02']
T2 = S['T2']['d02']
T4D = R['temp']
G4D = R['ghgt']

def temp_AGL(T,G,Z,z):
    "Compute mean T difference between 'z' meters above ground and T2 from model data (model ground level)."
    def tz(s,r):
        try:
            y = Z[s] + z
            x = pd.Series([ip.interp1d(G[s].loc[t], c, 'linear', bounds_error=False)(y) for t,c in r.iterrows()], index=r.index)
            return np.mean(T2[s] - x)
        except:
            return None

    return pd.Series(*zip(*[(tz(s,r),s) for s,r in T.iteritems()]))

# a = temp_AGL(T4D, G4D, Z, 50)

def landuse():
    with open('/home/arno/Documents/src/WRFV3/run/LANDUSE.TBL') as f:
        l = []
        for r in csv.reader(f):
            if len(r)==1:
                if r[0]=='SUMMER' or r[0]=='WINTER':
                    l[-1][1].append([r[0],[]])
                else:
                    l.append([r[0],[]])
            else:
                try:
                    if re.search('Unassigned',r[8]): continue
                    s = [float(x) for x in r[:8]]
                    s.append(r[8])
                    l[-1][1][-1][1].append(s)
                except: pass

    d = dict([(k,dict(v)) for k,v in l])

    modis = d['MODIFIED_IGBP_MODIS_NOAH']
    zs = dict([(x[0], x[4]) for x in modis['SUMMER']])
    zw = dict([(x[0], x[4]) for x in modis['WINTER']])

# Choi, Minha, Jennifer M. Jacobs, and William P. Kustas. “Assessment of Clear and Cloudy Sky Parameterizations for Daily Downwelling Longwave Radiation over Different Land Surfaces in Florida, USA.” Geophysical Research Letters 35, no. 20 (October 18, 2008). doi:10.1029/2008GL035731.
def cloud_N(Rs, Rso):
    """Cloud fraction from measured solar radiation and theoretical clear sky radiation.

    :param Rs: measured solar radiation
    :param Rso: clear sky solar radiation
    :returns: cloud fraction
    :rtype: float

    """
    return 1 - Rs / Rso

# Van Ulden, A. P., and A. A. M. Holtslag. “Estimation of Atmospheric Boundary Layer Parameters for Diffusion Applications.” Journal of Climate and Applied Meteorology 24, no. 11 (November 1985): 1196–1207. doi:10.1175/1520-0450(1985)024<1196:EOABLP>2.0.CO;2.
def LW_isothermal(Tr, N):
    return -5.67e-8 * Tr**4 * (1 - 9.35e-6 * Tr**2) + 60 * N

stationize(b)
Tr = T.apply(lambda c:c+a[c.name],0)
N = cloud_N(rs, Ra*b['b'])
Li = LW_isothermal(Tr, N)
