#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.signal as sig


def hann(df, start, width):
    w = (0.5 - 0.5 * np.cos(2 * np.pi * (df.index.to_series() - start) / width))
    print(len(w))
    return df * w, len(w)**.5 / np.sum(w)

class LS(object):
    def __init__(self, df, tshift=0):
        t = np.array(df.index, dtype='datetime64[m]').astype(int)
        period = np.timedelta64(1,'Y').astype('timedelta64[m]').astype(int)
        self._y = df.copy()
        self._y.index = t / period
        self._tf = tshift

    def tau(self, w):
        w2 = 2 * w
        t = np.r_['1,2', self._y.index].T
        wt = t * w2
        tau = 1 / w2 * np.arctan(np.sum(np.sin(wt), 0) / np.sum(np.cos(wt), 0))
        tp = t - tau
        self._F0 = 2**-.5 * np.exp(-1j * w * (self._tf - tau))
        self._wtp = pd.DataFrame(tp * w, index=self._y.index)

    def _lsdft(self, c, norm=1):
        y = c.dropna()
        wtp = self._wtp.loc[y.index].as_matrix()
        A = np.sum(np.cos(wtp)**2, 0) ** (-0.5)
        B = np.sum(np.sin(wtp)**2, 0) ** (-0.5)
        return norm * pd.Series(self._F0 * (A * y.dot(np.cos(wtp)) + 1j * B * y.dot(np.sin(wtp))))

    def raw(self, f=None):
        if f is not None:
            self.tau(2 * np.pi * f)
        x = self._y.apply(self._lsdft, 0)
        x.index = f # needs changing
        return x

    def welch(self, n, f=None):
        if f is not None:
            self.tau(2 * np.pi * f)
        def lsdft(c):
            print(c.name)
            y = c.dropna()
            start = y.index[0]
            T = y.index[-1] - start
            w = T / n
            def win(i):
                s = start+i*w/2
                return self._lsdft(*hann(y[(y.index>=s) & (y.index<=s+w)], s, w))
            return pd.concat([win(i) for i in range(2 * n - 1)], 1)
        x = pd.Panel(dict([(name, lsdft(col)) for name, col in self._y.iteritems()]))
        x.major_axis = f
        return x

    def coherence(self, n=8, f=None):
        G = self.welch(n, f)
        XY = (G.iloc[0] * np.conj(G.iloc[1])).mean(1)
        return np.abs(XY)**2 / (np.mean(G.iloc[0].abs()**2, 1) * np.mean(G.iloc[1].abs()**2, 1))



if __name__=="__main__":
    from astropy.stats import LombScargle
    N = 256
    s = 1
    rand = np.random.RandomState(42)
    t = np.linspace(0, 32, N)
    y = np.sin(2* np.pi * t) + rand.rand(N) * s

    f = np.fft.fftfreq(N, t[1]-t[0])
    i = (f>0)
    f = f[i]
    P = abs(np.fft.fft(y)[i]) ** 2 / N

    L = LombScargle(t, y).power(f, normalization='psd')
    f = np.linspace(0.1, 13, 500)
    # ls = LS(pd.DataFrame(y, index=t))
