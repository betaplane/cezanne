import pandas as pd
import xarray as xr
import scipy.signal as sig
from filters import Lanczos
import matplotlib.pyplot as plt
from cartopy import crs


with pd.HDFStore('../../data/CEAZAMet/station_data_new.h5') as S:
    p = S['pa_hpa'][['PLV', 'CNSD']].xs('prom', 1, 'aggr').resample('h').interpolate().dropna(0, 'any')['2016':]

slp = xr.open_dataarray('../../data/analyses/ERA/ERA-SLP-dailymean.nc')


class detect(object):
    def __init__(self, p, lowpass, extrema, slp):
        lw = Lanczos(p, lowpass)
        self.lp = p.rolling(**lw.roll).apply(lw)
        self.p = p
        self.plvmin, = sig.argrelmin(self.lp['PLV'].values.flatten(), order=extrema)
        self.cnsdmin, = sig.argrelmin(self.lp['CNSD'].values.flatten(), order=extrema)
        self.cnsdmax, = sig.argrelmax(self.lp['CNSD'].values.flatten(), order=extrema)
        self.t = pd.DatetimeIndex(pd.Series(self.lp.index + pd.Timedelta('4h')).dt.round('D'))
        self.slp = slp.copy()
        self.slp['time'] = pd.DatetimeIndex(pd.Series(slp.time).dt.round('D'))
        self.slp = self.slp - self.slp.mean('time')

    def plot(self):
        fig = plt.figure()
        plt.plot(self.p)
        self.pl = plt.plot(self.lp)
        plt.plot(self.lp['PLV'].iloc[self.plvmin], 'x')
        plt.plot(self.lp['CNSD'].iloc[self.cnsdmin], 'x')
        plt.plot(self.lp['CNSD'].iloc[self.cnsdmax], 'x')
        fig.show()

    def slp_plot(self, t):
        plt.set_cmap('PiYG_r')
        fig, axs = plt.subplots(2, 4, subplot_kw={'projection': crs.PlateCarree()}, figsize=(14, 6))
        for i in range(4):
            x = self.slp.sel(time = t + pd.Timedelta(i - 2, 'D'))
            pl = axs[0, i].contourf(x.lon, x.lat, x)
            axs[0, i].coastlines()
            plt.colorbar(pl, ax=axs[0, i])
            y = x.sel(lon=slice(-80, -65), lat=slice(-25, -40))
            pl = axs[1, i].contourf(y.lon, y.lat, y)
            axs[1, i].coastlines()
            plt.colorbar(pl, ax=axs[1, i])
        plt.pause(.1)


    def step(self):
        self.labels = []
        for i in self.plvmin:
            try:
                self.slp_plot(self.t[i])
                inp = input('(v)aguada, migratory (c)yclone, (m)ixed, (i)nconclusive, (q)uit: ')
                if inp == 'q':
                    break
                else:
                    self.labels.append((self.t[i], inp))
                plt.close()
            except:
                print('index not in SLP dataset')


if __name__ == '__main__':
    d = detect(p, '1D', 5*24, slp.sel(lon=slice(-120, -60), lat=slice(-15, -70)))
