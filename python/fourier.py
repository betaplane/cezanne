#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.signal as sig

def detrend(df):
    def lstsq(d):
        c = d.dropna()
        t = np.array(c.index, dtype='datetime64[m]').astype(int)
        b = np.linalg.lstsq(np.r_['0,2', np.ones(len(c)), t].T, c)[0]
        print('Column {}: mean {:.4E}, trend over all {:.4E}'.format(d.name, b[0], b[1]*(t[-1]-t[0])))
        return c - b[0] - b[1] * t
    return df.apply(lstsq, 0)

# Note:
# The normalization of the Lomb-Scarcle Discrete Fourier Transform (LSDFT) is such that
# abs(LSDFT)**2 ~ abs(numpy.fft.fft)**2 / N ~ scipy.signal.periodogram / 2

# The absolute value scales abs(LSDFT) ~ abs(numpy.fft.fft) / sqrt(N)
# This should be appropriate for averaging by Welch's method (see Welch, 1967 - specifically
# w.r.t the window normalization); but the result still needs to be divided by the total length
# of the input (not the windowed segments).


class LS(object):
    def __init__(self, df, detrend=False, tshift=0, period='Y', tol1=1e-4, tol2=1e-8):
        if detrend:
            self._y = detrend(df)
        else:
            self._y = df.copy()
        t = np.array(df.index, dtype='datetime64[m]').astype(int)
        period = np.timedelta64(1, period).astype('timedelta64[m]').astype(int)
        self._y.index = t / period
        self._tf = tshift
        self._tol1 = tol1
        self._tol2 = tol2

    @staticmethod
    def hann(df, start, width):
        w = (0.5 - 0.5 * np.cos(2 * np.pi * (df.index.to_series() - start) / width))
        print(len(w))
        return df * w, len(w) / np.sum(w)

    def set_freq(self, f):
        self._f = f
        i = (f!=0)
        self._i0 = np.where(f==0)[0]
        w = 2 * np.pi * f[i]
        w2 = 2 * w
        t = np.r_['1,2', self._y.index].T
        wt = t * w2
        tau = 1 / w2 * np.arctan(np.sum(np.sin(wt), 0) / np.sum(np.cos(wt), 0))
        tp = t - tau
        self._F0 = 2**-.5 * np.exp(-1j * w * (self._tf - tau))
        # self._F0 = 2**-.5
        wtp = tp * w
        self._cos = pd.DataFrame(np.cos(wtp), index=self._y.index)
        self._sin = pd.DataFrame(np.sin(wtp), index=self._y.index)
        self._cross = t * self._sin * self._cos

    def _lsdft(self, c, norm=1):
        y = c.dropna()
        ym = y.sum() * y.count()**-.5
        _cos = self._cos.loc[y.index].as_matrix()
        _sin = self._sin.loc[y.index].as_matrix()
        A = np.sum(_cos**2, 0) ** (-0.5)
        B = np.sum(_sin**2, 0)
        FTID = B**-.5 * y.dot(_sin)
        ssin2 = (B < self._tol1)
        cross = (np.sum(self._cross.loc[y.index].as_matrix(), 0) > self._tol2)
        print('denom {}, cross {}'.format(np.sum(ssin2), np.sum(cross)))
        FTID[ssin2] = ym
        FTID[ssin2 & cross] = 0
        dft = norm * self._F0 * (A * y.dot(_cos) + 1j * FTID)
        if len(self._i0):
            # NOTE: F0 is applied before re-inserting 0 frequency
            # in Scargle III (1989) FT(0) = sum(X) / sqrt(N) without the 1 / sqrt(2) factor
            dft = np.insert(dft, self._i0, ym)
        return pd.Series(dft, index=self._f)

    def raw(self, f=None):
        if f is not None:
            self.set_freq(f)
        return self._y.apply(self._lsdft, 0)

    def welch(self, n, f=None):
        if f is not None:
            self.set_freq(f)
        def lsdft(c):
            y = c.dropna()
            start = y.index[0]
            T = y.index[-1] - start
            w = T / n
            def win(i):
                s = start+i*w/2
                return self._lsdft(*self.hann(y[(y.index>=s) & (y.index<=s+w)], s, w))
            return pd.concat([win(i) for i in range(2 * n - 1)], 1)
        return pd.Panel(dict([(name, lsdft(col)) for name, col in self._y.iteritems()]))

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
