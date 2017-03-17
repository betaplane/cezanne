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
from astropy.stats import LombScargle
from scipy.integrate import quad
from datetime import datetime


def tx(d, freq='D'):
    return np.array(
        d.index, dtype='datetime64[{}]'.format(freq)).astype(
            float), d.as_matrix().flatten()


def LS(df):
    d = df.asfreq('1D')
    return pd.DataFrame(
        LombScargle(*tx(rf)).model(tx(d)[0], 1 / 365.24),
        index=d.index,
        columns=df.columns)


def tshift(df, delta):
    d = df.copy()
    d.index = pd.DatetimeIndex(d.index) + pd.Timedelta(delta)
    return d


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

    def a(self, h):
        "Shift hour angle to between [-pi, pi["
        return (h + np.pi) % (2 * np.pi) - np.pi

    def h_dec_dist(self, t):
        """
        Returns hour angle, sun's declination, sun-earth distance in AU.
        See https://en.wikipedia.org/wiki/Hour_angle.
        """
        self.date = t
        self.sun.compute(self)
        return self.a(self.sidereal_time() -
                      self.sun.ra), self.sun.dec, self.sun.earth_distance

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

    def all_params(self, t):
        self.date = t
        self.sun.compute(self)
        return self.a(
            self.sidereal_time() - self.sun.ra
        ), self.sun.dec, self.sun.earth_distance, self.sun.az


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
        return obs.S0 / np.pi * dist**-2 * (
            ws * np.sin(obs.lat) * np.sin(dec) + np.cos(obs.lat) * np.cos(dec)
            * np.sin(ws))

    return pd.DataFrame(
        dict([(c, per_station(c, r)) for c, r in stations.iterrows()]),
        index=index)


def ETRaDayN(station, days):
    "Numerical variant of ETRaDay."
    obs = observer(station)

    def mu(h, dec):
        return np.sin(obs.lat) * np.sin(dec) + np.cos(
            obs.lat) * np.cos(dec) * np.cos(h)

    def per_day(d):
        dec, dist = obs.decl(d)
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        return obs.S0 / np.pi * dist**-2 * quad(
            mu, 0, abs(ws), args=(dec, ))[0]

    return pd.DataFrame({station.name: [per_day(d) for d in days]}, index=days)


def ETRa1(stations, index):
    "Hourly extraterrestrial radiation based on explicit integral over hour angle interval given by Allen et al. 1998."
    t = pd.date_range(index[0], index[-1] + np.timedelta64(1, 'h'), freq='H')

    def per_station(code, station):
        print(code)
        obs = observer(station)
        # this is to account for the averaging period in the database
        te = t + np.timedelta64(4, 'h') - np.timedelta64(
            int(station.interval), 's')
        w, dec, dist = np.array([obs.h_dec_dist(i) for i in te]).T
        # sunset hour angle
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        # mask, where shift should ensure that intervals containing sunrise or -set are retained
        m = np.where(w > -ws, 1, 0)[1:] * np.where(w < ws, 1, 0)[:-1]
        # setting angles to sunset in between should compute correct integrals for partial hours at sunset/-rise
        # note thought that ws is recomputed throughout the night, so differences are not exactly zero - 
        # hence the mask
        w = np.where(w > -ws, w, -ws)
        w = np.where(w < ws, w, ws)
        # Note: this integral is for one hour
        # R = 12 / np.pi * S0 * (1 + 0.033 * np.cos(2 * np.pi / 365 * t[:-1].dayofyear)) * \
        mu_ave = 12 / np.pi * (np.diff(w) * np.sin(obs.lat) * np.sin(dec[1:]) + \
            np.cos(obs.lat) * np.cos(dec[1:]) * np.diff(np.sin(w)))
        R = obs.S0 * dist[1:]**-2 * mu_ave
        R[R < 0] = 0
        return {'Ra': R * m, 'mu': mu_ave * m, 'mask': m}

    # return per_station(*next(stations.iterrows()))
    p = pd.Panel(
        dict([(c, per_station(c, r)) for c, r in stations.iterrows()]))
    p.major_axis = index
    return p.transpose(2, 1, 0)


def ETRa2(stations, t):
    "Subdaily extraterrestrial radiation by using interval midpoint zenith computed directly bu PyEphem."

    def per_station(code, station):
        print(code)
        obs = observer(station)
        # here, shift the time vector by each station's averaging period so it contains the interval endpoints
        # add 1/2 hour to use midpoints (i.e. 270 mins together with UTC shift)
        te = t + np.timedelta64(270, 'm') - np.timedelta64(
            int(station.interval), 's')
        alt, d = np.array([obs.alt_dist(i) for i in te]).T
        # mu = cos zenith = sin alt
        mu = np.sin(alt)
        mu[mu < 0] = 0
        return obs.S0 * d**-2 * mu

    return pd.DataFrame(
        dict([(c, per_station(c, r)) for c, r in stations.iterrows()]),
        index=t)


def clear_sky(station, press):
    time = pd.date_range(start='2004-01-01', end=datetime.utcnow(), freq='H')
    print(station.name)
    obs = observer(station)

    def f(t):
        alt, d = obs.alt_dist(t)
        # mu = cos zenith = sin alt
        mu = np.sin(alt)
        if mu < 0: return 0
        Ra = obs.S0 * d**-2 * mu
        return Ra * np.exp(-0.00018 * press / mu)

    return pd.DataFrame(
        [f(t) for t in time], index=time - pd.Timedelta(4, 'H'))


# Lhomme, Vacher, and Rocheteau 2007
def clear_sky_param1(p, mu):
    "p in [hPa]"
    return np.exp(-1.8e-4 * p / mu)


# Meyers and Dale 1983 - without precipitable water and aerosol
def clear_sky_param2(p, mu):
    "p in [hPa]"
    # optical air mass
    m = 35 * (1224 * mu**2 + 1)**(-.5)
    return 1.021 - .084 * (m * (949e-6 * p * .051))**(-.5)


def clear_sky_I(station, time, param):
    """
    Numerical integration of <= daily intervals with different clear sky parameterizations.
    The logic hopefully works for arbitrary intervals, but it's not thouroughly thought through
    or tested.
    Need to use UTC. Index times of returned DataFrame are beginning of intervals.
    """
    # interval of integral in terms of hour angle
    dh = 2 * np.pi * time.freq.delta / pd.Timedelta('1D')
    print(station.name)
    obs = observer(station)

    def rad(h, lat, dec, dist):
        mu = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h)
        return obs.S0 * dist**-2 * mu * param(mu)

    def mu(h, H, lat, dec):
        return np.sin(lat) * np.sin(dec) + \
            np.cos(lat) * np.cos(dec) * (np.sin(H) - np.sin(h)) / dh

    def f(t):
        h, dec, dist, az = obs.all_params(t)
        ws = np.arccos(-np.tan(obs.lat) * np.tan(dec))
        H = obs.a(h + dh)
        R = 0
        mu_ave = 0
        # this tests if the interval includes the nightly jump from positive to negative hour angles
        # *and* if the endpoint of the interval is after sunrise
        # if yes, add the interval between sunrise and endpoint (the 'normal case' logic already computes
        # between startpoint and sunset)
        # in case of daily interval, this will be the whole day
        if H < h and H > -ws:
            R = quad(rad, -ws, min(ws, H), args=(obs.lat, dec, dist))[0] / dh
            mu_ave = mu(-ws, min(ws, H), obs.lat, dec)
        if H < -ws or h > ws:
            return R, az, mu_ave
        return R + quad(
            rad, max(-ws, h), min(ws, H), args=(obs.lat, dec, dist))[0] / dh, az, \
            mu_ave + mu(max(-ws, h), min(ws, H), obs.lat, dec)

    idx = pd.MultiIndex.from_product([[station.name], ['Rso', 'az', 'mu']])
    return pd.DataFrame([f(t) for t in time], index=time, columns=idx)


def Rso1(Ra, z):
    "Daily clear sky only."
    return (.75 + 2e-5 * z) * Ra


def Rso2(Ra, mu, p):
    "Hourly clear sky, p in hPa"
    return Ra * np.exp(-0.00018 * p / mu)


def locMax(df, window):
    ro = df.rolling(window).apply(np.argmax)
    i = np.array(ro.index,dtype='datetime64[D]').flatten() + \
        np.array(ro.as_matrix() - window + 1,dtype='timedelta64[D]').flatten()
    return pd.Index(set(i)).dropna().sort_values()


def locMaxDay(RaDay, rs, mask):
    d = rs.copy()
    # first, set NaNs at night to zero so as to not inflate daily averages
    d[d.isnull()] = mask[mask == 0]
    # then, zero out not-NaN night values
    dm = (d * mask).groupby(d.index.date).mean()
    dm[dm == 0] = np.nan
    ratio = dm / RaDay
    ratio = ratio[ratio < 1].asfreq('1D')
    return dm, ratio.loc[locMax(ratio, 10)]


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
            b.append((a, (err / n)**.5, n))
    return pd.DataFrame(b, index=rm.columns, columns=['b', 'rms', 'n'])


def quality():
    b = regression(rs, Ra).sort_index()
    # manual selection of sensors at stations which have 2 - for now take the ones with lower RMS
    from CEAZAMet import mult
    c = b.loc[mult(rs)]
    drop_codes = [
        '28', 'ANDACMP10', 'CACHRSTM', 'COMBRS', 'INILLARS', 'MINRS', 'PCRS2',
        'TLHRSTM'
    ]
    b.drop(drop_codes, level='code', inplace=True)
    rs.drop(drop_codes, level='code', inplace=True, axis=1)

    # very high regression coefficients are also suspicious
    b.sort_values('b', inplace=True)
    drop_codes.extend(['INIA66', 'PAGN', 'LMBT', 'SALH', ''])


def detect_breaks(df):
    i = df.dropna().index
    return pd.Series(np.diff(i), index=i[1:]).sort_values()


def residuals(code):
    return pd.concat(
        (b['b'][code][0] * Ra[code], rs[code]), axis=1).diff(
            1, axis=1).iloc[:, 1]


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


def temp_AGL(T, G, Z, z):
    "Compute mean T difference between 'z' meters above ground and T2 from model data (model ground level)."

    def tz(s, r):
        try:
            y = Z[s] + z
            x = pd.Series(
                [
                    ip.interp1d(G[s].loc[t], c, 'linear',
                                bounds_error=False)(y) for t, c in r.iterrows()
                ],
                index=r.index)
            return np.mean(T2[s] - x)
        except:
            return None

    return pd.Series(*zip(* [(tz(s, r), s) for s, r in T.iteritems()]))


# a = temp_AGL(T4D, G4D, Z, 50)


def landuse():
    with open('/home/arno/Documents/src/WRFV3/run/LANDUSE.TBL') as f:
        l = []
        for r in csv.reader(f):
            if len(r) == 1:
                if r[0] == 'SUMMER' or r[0] == 'WINTER':
                    l[-1][1].append([r[0], []])
                else:
                    l.append([r[0], []])
            else:
                try:
                    if re.search('Unassigned', r[8]): continue
                    s = [float(x) for x in r[:8]]
                    s.append(r[8])
                    l[-1][1][-1][1].append(s)
                except:
                    pass

    d = dict([(k, dict(v)) for k, v in l])

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

if __name__ == "__main__":
    # solar noon offset from UTC is around 4h 45 min in the area
    Noon = pd.Timedelta(-285, 'm')

    D = pd.HDFStore('../../data/tables/station_data_new.h5')
    rs = D['rs_w'].xs('prom', level='aggr', axis=1)
    R = pd.HDFStore('../../data/tables/station_data_raw.h5')
    rr = R['rs_w'].xs('avg', level='aggr', axis=1)
    # T = hh.extract(D['ta_c'], 'prom', C2K=True)

    sta = D['sta']

    r = hh.stationize(rs.loc[:, '5':'5'].xs('2', level='elev', axis=1))
    Rm = ETRaDay(sta.loc['5':'5'],
                 pd.date_range('2004-01-01T12', '2017-03-01T12', freq='D'))
    Ra = ETRa1(sta.loc['5':'5'], rs.index)
    # Ra = D['Ra_w']
    m = Ra['mask']

    P = pd.HDFStore('../../data/tables/pressure.h5')
    pm = P['p']['5'].mean()
    st = sta.loc['5']

    # b0,b1 = P['fit']
    # z = sta.loc['5']['elev']
    # p = np.exp(b0 + b1*z)
    # p2 = 1013 * (1 - 0.0065 * z / 293) ** 5.26

    # cs1 = Rso1(Rm, z)
    cs2 = tshift(Rso2(Ra['Ra'], Ra['mu'], pm), '30m')

    hours = pd.date_range(start='2004-01-01', end=datetime.utcnow(), freq='H')
    days = pd.date_range(start='2004-01-01', end=datetime.utcnow(), freq='D')
    csI = clear_sky_I(st, hours, partial(clear_sky_param1, pm))
    csI2 = clear_sky_I(st, hours, partial(clear_sky_param2, pm))
    csId = clear_sky_I(st, days, partial(clear_sky_param1, pm))

    D = pd.DataFrame(index=hours)
    for n, s in sta.iterrows():
        try:
            p = press[n]
        except:
            p = 1013 * (1 - 0.0065 * s.elev / 293)**5.26
        df = clear_sky_I(s, hours, partial(clear_sky_param1, p))
        D = pd.concat((D, df), axis=1)

# rm, ra = locMaxDay(Rm, r, m)
# rm = tshift(rm, '12H')

# S = pd.HDFStore('../../data/tables/LinearLinear.h5')
# R = pd.HDFStore('../../data/tables/4Dfields.h5')
# Z = S['z']['d02']
# T2 = S['T2']['d02']
# T4D = R['temp']
# G4D = R['ghgt']
