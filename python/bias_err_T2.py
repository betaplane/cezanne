#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap
from matplotlib.figure import SubplotParams
import helpers as hh
from functools import partial
from astropy.stats import LombScargle

K = 273.15
dd = lambda s: os.path.join('../data', s)

D = pd.HDFStore(dd('station_data.h5'))

sta = D['sta']
T = hh.extract(D['ta_c'], 'prom') + K

S = pd.HDFStore(dd('LinearLinear.h5'))
Tm = S['T2']
Z = S['z']

# b = (T-Tm.minor_xs('d02')).dropna(0,'all')


def lsq(d):
    y = d.dropna()
    if y.empty: return np.nan
    x = np.r_['0,2', np.array(y.index).astype('datetime64[h]').astype(float)]
    return hh.lsq(y.as_matrix(), x.T)['b1']


# b.apply(lsq,0)
# 
# t = T['3'].dropna().asfreq('1H','pad')
# tf = np.fft.rfft(t)


def ff(x):
    y = 24 * 365.25
    n = int(np.floor(len(x) / y))
    ft = np.fft.rfft(x, int(n * y))
    ft[n] = 0
    return np.fft.irfft(ft)


def smap(x):
    m = max(abs(x))
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-m, vmax=m))
    sm.set_array(x)
    return sm


ma = hh.basemap()

z = {
    'd01': 'd01',
    'd02': 'd02',
    'd03_0_00': 'd03_op',
    'd03_0_12': 'd03_op',
    'd03_orl': 'd03_orl',
    'fnl': 'fnl'
}


# fig, axes = plt.subplots(nrows=3,ncols=4,figsize=(7,6))
def bias_maps():
    """
top row: height differences, station-model
middle row: bias, model-station
bottom row: bias, adjusted for 6.5K/km lapse rate
	"""


fig = plt.figure(
    figsize=(8, 8),
    subplotpars=SubplotParams(left=0.08, right=0.86, wspace=0.06, hspace=0.04))
for k in range(3):
    for j, x in enumerate(Tm.minor_axis):
        ax = plt.subplot(3, 4, k * 4 + j + 1)

        B = (Tm.minor_xs(x) - T).dropna(0, 'all')
        b = B.mean().dropna()
        dz = sta['elev'] - Z[x]
        if k == 0:
            ax.set_title(x)
            D = pd.concat(
                (sta.loc[b.index, ('lon', 'lat')], dz), axis=1).dropna()
            m = 1750
        elif k == 1:
            D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
            m = 13
        else:
            D = pd.concat(
                (sta.loc[:, ('lon', 'lat')], b - 0.0065 * dz), axis=1).dropna()
        sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-m, vmax=m))
        sm.set_array(D.iloc[:, -1])
        sm.set_cmap('coolwarm')
        for i, a in D.iterrows():
            map.plot(
                a['lon'],
                a['lat'],
                'o',
                color=sm.to_rgba(a.iloc[-1]),
                latlon=True)
        map.drawcoastlines()
        map.drawparallels(range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
        map.drawmeridians(range(-72, -69, 1), labels=[0, 0, 0, max(k - 1, 0)])

        if j == 3:
            bb = ax.get_position()
            plt.colorbar(
                sm,
                cax=fig.add_axes([bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))


def bias_maps2():
    fig = plt.figure(
        figsize=(8, 8),
        subplotpars=SubplotParams(
            left=0.08, right=0.86, wspace=0.06, hspace=0.06))
    cols = 4
    for j, x in enumerate(['d02', 'd03_orl', 'd03_0_00', 'd03_0_12']):
        B = (Tm[x] - T).dropna(0, 'all')
        b = B.mean().dropna()
        dz = sta['elev'] - Z[z[x]]
        print('{} {}'.format(np.max(abs(b)), np.max(abs(dz))))
        for k in range(3):
            ax = plt.subplot(3, cols, k * cols + j + 1)
            if k == 0:
                ax.set_title(x)
                D = pd.concat(
                    (sta.loc[b.index, ('lon', 'lat')], dz), axis=1).dropna()
                m = 1150
            elif k == 1:
                D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
                m = 10
            else:
                D = pd.concat(
                    (sta.loc[:, ('lon', 'lat')], b - 0.0065 * dz),
                    axis=1).dropna()
                m = 10
            plt.set_cmap('coolwarm')
            lon, lat, c = D.as_matrix().T
            p = ma.scatter(lon, lat, c=c, latlon=True)
            p.set_clim((-m, m))
            ma.drawcoastlines()
            ma.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            ma.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 1, 0)])

            if j == cols - 1:
                bb = ax.get_position()
                plt.colorbar(
                    p,
                    cax=fig.add_axes(
                        [bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))
    fig.show()


fig = plt.figure(
    figsize=(10, 8),
    subplotpars=SubplotParams(left=0.08, right=0.86, wspace=0.06, hspace=0.06))
cols = 5
day = lambda x: x.groupby(x.index.date).min()
m = 10
for j, x in enumerate(['fnl', 'd01', 'd02', 'd03_orl', 'd03_0_00']):
    dz = sta['elev'] - Z[z[x]]
    t = Tm[x] - 0.0065 * dz
    B = Tm[x] - (Tm[x] - T).dropna(0, 'all').mean()
    dt = day(T)
    for k in range(3):
        ax = plt.subplot(3, cols, k * cols + j + 1)
        if k == 0:
            ax.set_title(x)
            b = (day(Tm[x]) - dt).mean().dropna()
            D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
        elif k == 1:
            b = (day(t) - dt).mean().dropna()
            D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
        else:
            b = (day(B) - dt).mean().dropna()
            D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
        print(abs(b).max())
        plt.set_cmap('coolwarm')
        lon, lat, c = D.as_matrix().T
        p = ma.scatter(lon, lat, c=c, latlon=True)
        p.set_clim((-m, m))
        ma.drawcoastlines()
        ma.drawparallels(range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
        ma.drawmeridians(range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
        # 		plt.colorbar(p)
        if j == cols - 1:
            bb = ax.get_position()
            plt.colorbar(
                p,
                cax=fig.add_axes([bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))
fig.show()


def rms_plot():
    """
top row: total rms
middle row: rms with 6.5K/km lapse rate applied
bottom row: rms, bias removed
	"""
    fig = plt.figure(
        figsize=(10, 8),
        subplotpars=SubplotParams(
            left=0.08, right=0.86, wspace=0.06, hspace=0.04))
    for k in range(3):
        for j, x in enumerate(Tm.minor_axis):
            ax = plt.subplot(3, 4, k * 4 + j + 1)

            B = (Tm.minor_xs(x) - T).dropna(0, 'all')
            dz = sta['elev'] - Z[x]

            if k == 0:
                ax.set_title(x)
                b = ((B**2).mean()**.5).dropna()
                D = pd.concat(
                    (sta.loc[b.index, ('lon', 'lat')], b), axis=1).dropna()
            elif k == 1:
                b = (((B - 0.0065 * dz)**2).mean()**.5).dropna()
                D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
            else:
                b = B.std().dropna()
                D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
            sm = cm.ScalarMappable(norm=colors.Normalize(
                vmin=b.min(), vmax=b.max()))
            sm.set_array(D.iloc[:, -1])
            sm.set_cmap('gnuplot')
            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 1, 0)])
            plt.colorbar(sm)


def mae_maps():
    cols = ['fnl', 'd01', 'd02', 'd03_orl', 'd03_0_00', 'd03_0_12']
    ncols = len(cols)
    cmin = np.zeros((3, ncols))
    cmax = cmin.copy()
    p = [[], [], []]
    fig, axs = plt.subplots(
        3,
        ncols,
        figsize=(10, 8),
        subplotpars=SubplotParams(
            left=0.08, right=0.86, wspace=0.06, hspace=0.04))
    for j, x in enumerate(cols):
        B = (Tm[x] - T).dropna(0, 'all')
        dz = sta['elev'] - Z[z[x]]
        axs[0, j].set_title(x)
        for k in range(3):
            plt.sca(axs[k, j])
            if k == 0:
                b = abs(B).mean().dropna()
                D = pd.concat(
                    (sta.loc[b.index, ('lon', 'lat')], b), axis=1).dropna()
            elif k == 1:
                b = abs(B - 0.0065 * dz).mean().dropna()
                D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()
            else:
                b = abs(B - B.mean()).mean().dropna()
                D = pd.concat((sta.loc[:, ('lon', 'lat')], b), axis=1).dropna()

            cmin[k, j] = min(b)
            cmax[k, j] = max(b)
            print('{} {}'.format(b.max(), b.min()))
            lon, lat, c = D.as_matrix().T
            plt.set_cmap('gnuplot')
            p[k].append(ma.scatter(lon, lat, c=c, latlon=True))
            ma.drawcoastlines()
            ma.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            ma.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 1, 0)])
            if j == ncols - 1:
                for i in range(ncols):
                    p[k][i].set_clim((min(cmin[k, :]), max(cmax[k, :])))
                bb = axs[k, j].get_position()
                plt.colorbar(
                    p[k][j],
                    cax=fig.add_axes(
                        [bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))
    fig.show()


def rms_seasonal():
    fig = plt.figure(
        figsize=(9, 8),
        subplotpars=SubplotParams(
            left=0.08, right=0.86, wspace=0.06, hspace=0.04))
    for k in range(4):
        for j, x in enumerate(Tm.minor_axis):
            ax = plt.subplot(4, 4, k * 4 + j + 1)
            if k == 0: ax.set_title(x)

            B = (Tm.minor_xs(x) - T).dropna(0, 'all')

            # total rms
            # 		b = (B**2).groupby(lambda idx: int((idx.month%12)/3))

            # mean bias removed
            # 		b = ((B-B.mean())**2).groupby(lambda idx: int((idx.month%12)/3))

            # seasonal bias removed
            b = B.groupby(lambda idx: int((idx.month % 12) / 3)).get_group(k)
            c = ((b - b.mean())**2).mean()**.5

            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()

            sm = cm.ScalarMappable(norm=colors.Normalize(
                vmin=c.min(), vmax=c.max()))
            sm.set_array(D.iloc[:, -1])
            sm.set_cmap('gnuplot')
            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
            plt.colorbar(sm)


def cycle_bias():
    fig = plt.figure(
        figsize=(12, 8),
        subplotpars=SubplotParams(
            left=0.06, right=0.96, wspace=0.06, hspace=0.04))
    for k in range(3):
        for j in range(5):
            ax = plt.subplot(3, 5, k * 5 + j + 1)
            if j == 0: B = T
            else:
                x = Tm.minor_axis[j - 1]
                B = (Tm.minor_xs(x) - T).dropna(0, 'all')
            if k == 0:
                ax.set_title('obs') if j == 0 else ax.set_title(x)
                b = B.groupby(B.index.month).mean()
                c = b.max() - b.min()
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')
            elif k == 1:
                b = B.groupby(B.index.month).std()
                c = b.max() - b.min()
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')
            else:
                b = B.groupby(B.index.month).mean()
                c = b.idxmax().dropna().astype(int)
                sm = cm.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=12))
                sm.set_cmap('hsv')

            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()
            sm.set_array(D.iloc[:, -1])

            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
            plt.colorbar(sm)


def cycles():
    fig = plt.figure(
        figsize=(10, 8),
        subplotpars=SubplotParams(
            left=.1, right=.96, bottom=.06, top=.92, wspace=.1, hspace=.12))
    for k in range(4):
        for j in range(5):
            ax = plt.subplot(4, 5, k * 5 + j + 1)
            if j == 0: B = T
            else:
                x = Tm.minor_axis[j - 1]
                B = Tm.minor_xs(x).dropna(0, 'all')
            if k == 0:
                ax.set_title('obs') if j == 0 else ax.set_title(x)
                b = B.groupby(B.index.month).mean()
                c = b.max() - b.min()
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')
            elif k == 1:
                b = B.groupby(B.index.month).mean()
                c = b.idxmax().dropna().astype(int)
                sm = cm.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=12))
                sm.set_cmap('hsv')
            elif k == 2:
                b = B.groupby(B.index.month).std()
                c = b.mean()
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')
            else:
                b = B.groupby(B.index.month).std()
                c = b.max() - b.min()
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')

            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()
            sm.set_array(D.iloc[:, -1])

            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
            plt.colorbar(sm)


def pow(T, d):
    try:
        c = d.dropna()
        t = np.array(c.index, dtype='datetime64[h]').astype(float)
        if max(t) - min(t) < T / 4: return np.nan
        x = c.as_matrix()
        y = LombScargle(t, x).model(np.linspace(0, T, 100), 1 / T)
        return max(y) - min(y)
    except:
        return np.nan


def daily(d):
    c = d.groupby(d.index.date)
    return (c.max() - c.min()).mean()


def period():
    fig = plt.figure(
        figsize=(12, 6),
        subplotpars=SubplotParams(
            left=.04, right=.96, bottom=.06, top=.92, wspace=.1, hspace=.12))
    for k in range(2):
        for j in range(5):
            ax = plt.subplot(2, 5, k * 5 + j + 1)
            if j == 0: B = T
            else:
                x = Tm.minor_axis[j - 1]
                B = Tm.minor_xs(x).dropna(0, 'all')
            if k == 0:
                ax.set_title('obs') if j == 0 else ax.set_title(x)
                c = B.apply(partial(pow, 1 / (24 * 365.25)))
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')
            elif k == 1:
                c = B.apply(partial(pow, 1 / 24))
                sm = cm.ScalarMappable(norm=colors.Normalize(
                    vmin=c.min(), vmax=c.max()))
                sm.set_cmap('gnuplot')

            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()
            sm.set_array(D.iloc[:, -1])

            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
            plt.colorbar(sm)


def period_rescaled():
    fig = plt.figure(
        figsize=(12, 6),
        subplotpars=SubplotParams(
            left=.04, right=.9, bottom=.06, top=.92, wspace=.1, hspace=.12))
    for k in range(2):
        for j in range(5):
            ax = plt.subplot(2, 5, k * 5 + j + 1)
            if j == 0: B = T
            else:
                x = Tm.minor_axis[j - 1]
                B = Tm.minor_xs(x).dropna(0, 'all')
            if k == 0:
                ax.set_title('obs') if j == 0 else ax.set_title(x)
                c = B.apply(partial(pow, 1 / (24 * 365.25)))
                sm = cm.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=12))
            elif k == 1:
                c = B.apply(partial(pow, 1 / 24))
                sm = cm.ScalarMappable(norm=colors.Normalize(vmin=1, vmax=16))

            sm.set_cmap('gnuplot')

            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()
            sm.set_array(D.iloc[:, -1])

            for i, a in D.iterrows():
                map.plot(
                    a['lon'],
                    a['lat'],
                    'o',
                    color=sm.to_rgba(a.iloc[-1]),
                    latlon=True)
            map.drawcoastlines()
            map.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            map.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])

            if j == 4:
                bb = ax.get_position()
                plt.colorbar(
                    sm,
                    cax=fig.add_axes(
                        [bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))


def cycle_maps():
    cols = ['obs', 'fnl', 'd01', 'd02', 'd03_orl', 'd03_0_00', 'd03_0_12']
    ncols = len(cols)
    cmin = np.zeros((2, ncols))
    cmax = cmin.copy()
    fig, axs = plt.subplots(
        2,
        ncols,
        figsize=(12, 6),
        subplotpars=SubplotParams(
            left=.04, right=.9, bottom=.06, top=.92, wspace=.1, hspace=.12))
    plt.set_cmap('gnuplot')
    p = [[], []]
    for j, x in enumerate(cols):
        if x == 'obs': B = T
        else:
            B = Tm[x].dropna(0, 'all')
        axs[0, j].set_title(x)

        for k in range(2):
            if k == 0:
                c = B.apply(partial(pow, 24 * 365.25))
            elif k == 1:
                # 			c = B.apply(partial(pow,24))
                c = daily(B)
            cmin[k, j] = min(c)
            cmax[k, j] = max(c)
            plt.sca(axs[k, j])
            D = pd.concat(
                (sta.loc[c.index, ('lon', 'lat')], c), axis=1).dropna()
            lon, lat, col = D.as_matrix().T
            p[k].append(ma.scatter(lon, lat, c=col, latlon=True))
            ma.drawcoastlines()
            ma.drawparallels(
                range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
            ma.drawmeridians(
                range(-72, -69, 1), labels=[0, 0, 0, max(k - 2, 0)])
            if j == ncols - 1:
                for i in range(ncols):
                    p[k][i].set_clim((min(cmin[k, :]), max(cmax[k, :])))
                bb = axs[k, j].get_position()
                plt.colorbar(
                    p[k][j],
                    cax=fig.add_axes(
                        [bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))


def land_sea():
    lm = S['land_mask']
    fig = plt.figure(figsize=(12, 6))
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1))
    sm.set_cmap('gnuplot')
    for j in range(5):
        ax = plt.subplot(1, 5, j + 1)
        ax.set_title(lm.columns[j])
        D = pd.concat((sta.loc[:, ('lon', 'lat')], lm.iloc[:, j]), axis=1)
        sm.set_array(D.iloc[:, -1])
        for i, a in D.iterrows():
            map.plot(
                a['lon'],
                a['lat'],
                'o',
                color=sm.to_rgba(a.iloc[-1]),
                latlon=True)
        map.drawcoastlines()
        map.drawparallels(range(-32, -28, 1), labels=[max(1 - j, 0), 0, 0, 0])
        map.drawmeridians(range(-72, -69, 1), labels=[0, 0, 0, 1])
        if j == 4:
            bb = ax.get_position()
            plt.colorbar(
                sm,
                cax=fig.add_axes([bb.x1 + 0.02, bb.y0, 0.02, bb.y1 - bb.y0]))
