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


def stationize(df):
    try:
        return pd.DataFrame(df, columns=df.columns.get_level_values('station'))
    except:
        return pd.DataFrame(df, index=df.index.get_level_values('station'))


class observer(ep.Observer):
    # https://en.wikipedia.org/wiki/Solar_constant
    S0 = 1361
    rad = np.pi / 180
    def __init__(self, station):
        super(ep.Observer, self).__init__()
        self.sun = ep.Sun()
        self.lon = station.lon * self.rad
        self.lat = station.lat * self.rad
        self.elevation = station.elev

    def allparams(self, t):
        """
        Returns hour angle, sun's declination, sun-earth distance in AU.
        See https://en.wikipedia.org/wiki/Hour_angle.
        """
        self.date = t
        self.sun.compute(self)
        return self.sidereal_time() - self.sun.ra, self.sun.dec, self.sun.earth_distance

    def alt_dist(self, t):
        """
        Returns sun's zenith angle based on PyEphem Observer and time, and as convenience also the
        earth-sun distance in AU
        See http://rhodesmill.org/pyephem/quick.html#bodies.
        """
        self.date = t
        self.sun.compute(self)
        return self.sun.alt, self.sun.earth_distance

    def decl(self, t):
        self.date = t
        self.sun.compute(self)
        return self.sun.dec, self.sun.earth_distance

    def angle_decl(self, t):
        self.date = t
        self.sun.compute(self)
        return self.sidereal_time() - self.sun.ra, self.sun.dec


# extra-terrestrial radiation
# Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. FAO Irrigation and drainage paper No. 56.
# Rome: Food and Agriculture Organization of the United Nations 56, 97–156.

def ETRaDay(stations, index):
    "Daily average extraterrestrial radiation from explicit integral over hour angle (as in Allen et al.)."
    def per_station(code, station):
        print(code)
        obs = observer(station)
        dec, dist = np.array([obs.decl(i) for i in index]).T
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        return 1/np.pi * obs.S0 * dist**-2 * (ws * np.sin(obs.lat) * np.sin(dec) + np.cos(dec) * np.sin(ws))
    return pd.DataFrame(dict([(c, per_station(c, r)) for c,r in stations.iterrows()]), index=index)


def ETRa1(stations, index):
    "Subdaily extraterrestrial radiation based on explicit integral over hour angle interval given by Allen et al. 1998."
    t = pd.date_range(index[0], index[-1] + np.timedelta64(1, 'h'), freq='H')
    def per_station(code, station):
        print(code)
        obs = observer(station)
        # this is to account for the averaging period in the database
        te = t + np.timedelta64(4, 'h') - np.timedelta64(int(station.interval), 's')
        w, dec, dist = np.array([obs.allparams(i) for i in te]).T
        w[w > np.pi] -= 2 * np.pi
        w[w < -np.pi] += 2 * np.pi
        # sunset hour angle
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        # mask, where shift should ensure that intervals containing sunrise or -set are retained
        m = np.where(w > -ws, 1, 0)[1:] * np.where(w < ws, 1, 0)[:-1]
        # setting angles to sunset in between should compute correct integrals for partial hours at sunset/-rise
        # note thought that ws is recomputed throughout the night, so differences are not exactly zero - 
        # hence the mask
        w = np.where(w > -ws, w, -ws)
        w = np.where(w < ws, w, ws)
        # R = 12 / np.pi * S0 * (1 + 0.033 * np.cos(2 * np.pi / 365 * t[:-1].dayofyear)) * \
        R = 12 / np.pi * obs.S0 * dist[1:]**-2 * \
        (np.diff(w) * np.sin(obs.lat) * np.sin(dec[1:]) + np.cos(obs.lat) * np.cos(dec[1:]) * np.diff(np.sin(w)))
        R[R < 0] = 0
        return {'Ra':R*m, 'mask':m}
    # return per_station(*next(stations.iterrows()))
    p = pd.Panel(dict([(c, per_station(c,r)) for c,r in stations.iterrows()]))
    p.major_axis = index
    return p


def ETRa2(stations, t):
    "Subdaily extraterrestrial radiation by using interval midpoint zenith computed directly bu PyEphem."
    def per_station(code, station):
        print(code)
        obs = observer(station)
        # here, shift the time vector by each station's averaging period so it contains the interval endpoints
        # add 1/2 hour to use midpoints (i.e. 270 mins together with UTC shift)
        te = t + np.timedelta64(270, 'm') - np.timedelta64(int(station.interval), 's')
        alt, d = np.array([obs.alt_dist(i) for i in te]).T
        # mu = cos zenith = sin alt
        mu = np.sin(alt)
        mu[mu<0] = 0
        return obs.S0 * d**-2 * mu
    return pd.DataFrame(dict([(c, per_station(c,r)) for c,r in stations.iterrows()]), index=t)


def mask_night(stations, index):
    t = pd.date_range(index[0], index[-1] + np.timedelta64(1, 'h'), freq='H')
    def per_station(code, station):
        print(code)
        obs = observer(station)
        # this is to account for the averaging period in the database
        te = t + np.timedelta64(4, 'h') - np.timedelta64(int(station.interval), 's')
        w, dec = np.array([obs.angle_decl(i) for i in te]).T
        w[w > np.pi] -= 2 * np.pi
        w[w < -np.pi] += 2 * np.pi
        # sunset hour angle
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        # before sunrise
        # this shifts the sunrise index to one earlier, so that the interval containing the sunrise is retained
        m[:-1] = np.logical_or(m[:-1], m[1:])
        # after sunset
        return m * np.where(w < ws, 1, 0)[:-1]
    return pd.DataFrame(dict([(c, per_station(c,r)) for c,r in stations.iterrows()]), index=index)


Rm = ETRaDay(sta, rs.index)
Ra = ETRa1(sta, rs.index)
# Ra = D['Ra_w']

def Rso1(Ra, z):
    return (.75 + 2e-5 * z) * Ra


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

def quality():
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


# S = pd.HDFStore('../../data/tables/LinearLinear.h5')
# R = pd.HDFStore('../../data/tables/4Dfields.h5')
# Z = S['z']['d02']
# T2 = S['T2']['d02']
# T4D = R['temp']
# G4D = R['ghgt']

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

# stationize(b)
# Tr = T.apply(lambda c:c+a[c.name],0)
# N = cloud_N(rs, Ra*b['b'])
# Li = LW_isothermal(Tr, N)
