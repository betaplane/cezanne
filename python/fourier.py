#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import chi2
from functools import partial

def detrend(df):
    def lstsq(d):
        c = d.dropna()
        t = np.array(c.index, dtype='datetime64[m]').astype(int)
        b = np.linalg.lstsq(np.r_['0,2', np.ones(len(c)), t].T, c)[0]
        print('Column {}: mean {:.4E}, trend over all {:.4E}'.format(d.name, b[0], b[1]*(t[-1]-t[0])))
        return c - b[0] - b[1] * t
    return df.apply(lstsq, 0)

# Schulz, Michael, and Karl Stattegger. “SPECTRUM: Spectral Analysis of Unevenly Spaced Paleoclimatic Time Series.” Computers & Geosciences 23, no. 9 (1997): 929–945.


class LS(object):
    """
    Note on normalization:
    The normalization corresponds to a FFT with 1/sqrt(N).
    The following should be equivalent (for norm='psd'):
    > abs(LSDFT)**2 ~ abs(numpy.fft.fft)**2 / N / f_s, where f_s is sample frequency
    > 2 * abs(LSDFT)**2 ~ scipy.signal.periodogram(fs=f_s) with argument fs given
    > abs(LSDFT)**2 ~ astropy.stats.LombScargle(normalization='psd') / f_s
    """
    def __init__(self, df, detrend=False, tshift=0, period='Y', tol1=1e-4, tol2=1e-8):
        self._y = df.copy()
        t = np.array(df.index, dtype='datetime64[m]').astype(int)
        period = np.timedelta64(1, period).astype('timedelta64[m]').astype(int)
        self._y.index = t / period
        self._tf = tshift # shift should be applied to one of the time series, see Schulz and Stattegger
        self._tol1 = tol1
        self._tol2 = tol2
        self._detr = detrend

    def set_freq(self, f):
        # throw out 0 frequency - inconsistent when windows are used and best computed as average anyway
        self._f = f[f!=0]
        w = 2 * np.pi * self._f
        w2 = 2 * w
        t = self._y.index.values.reshape((-1,1)) # nObs x 1
        wt = t * w2                              # nObs x nFreqs
        tau = 1 / w2 * np.arctan2(np.sum(np.sin(wt), 0), np.sum(np.cos(wt), 0)) # nFreqs x 0
        tp = t - tau # nObs x nFreqs
        self._F0 = 2**-.5 * np.exp(-1j * w * (self._tf - tau)) # nFreqs x 0
        # self._F0 = 2**-.5
        wtp = tp * w # nObs x nFreqs
        self._cos = pd.DataFrame(np.cos(wtp), index=self._y.index)
        self._sin = pd.DataFrame(np.sin(wtp), index=self._y.index)
        self._cross = t * self._sin * self._cos

    @staticmethod
    def _detrend(c):
        t = np.array(c.index, dtype='datetime64[m]').astype(int)
        b = np.linalg.lstsq(np.r_['0,2', np.ones(len(c)), t].T, c)[0]
        print('Column {}: mean {:.4E}, trend over all {:.4E}'.format(c.name, b[0], b[1]*(t[-1]-t[0])))
        return c - b[0] - b[1] * t

    # can be applied either to whole column (with norm) or window (without norm -
    # it's applied by the window function)
    def _lsdft(self, c, norm=None):
        y = c.dropna()
        if self._detr:
            y = self._detrend(y)
        ym = y.sum() * y.count()**-.5
        _cos = self._cos.loc[y.index].as_matrix()
        _sin = self._sin.loc[y.index].as_matrix()
        A = np.sum(_cos**2, 0) ** (-0.5)
        B = np.sum(_sin**2, 0)
        # The variable names are the same as in Scargle's (1989) algorithm (paper III)
        # These are his suggestions for weeding out possible undefined points.
        FTID = B**-.5 * y.dot(_sin)
        ssin2 = (B < self._tol1)
        cross = (np.sum(self._cross.loc[y.index].as_matrix(), 0) > self._tol2)
        if (np.sum(ssin2)):
            print('denom {}, cross {}'.format(np.sum(ssin2), np.sum(cross)))
        FTID[ssin2] = ym
        FTID[ssin2 & cross] = 0
        dft = self._F0 * (A * y.dot(_cos) + 1j * FTID)

        # the multiplication by sqrt(N) is baked into the normalization for
        # full-length windows here
        if norm=='psd':
            dft = dft * ((y.index[-1] - y.index[0]) / len(y)) ** .5
        elif norm=='pow':
            dft = dft / len(y) ** .5
        return pd.Series(dft, index=self._f)

    def raw(self, f=None, norm='psd'):
        if f is not None:
            self.set_freq(f)
        return self._y.apply(self._lsdft, 0, norm=norm)

    @staticmethod
    def hann(t):
        return (0.5 - 0.5 * np.cos(2 * np.pi * t))

    def _window(self, df, start, width, norm, i):
        """
        Applies windowed LombScargle DFT with the following choice of normalization
        (still needs to be squared and multiplied by 2 afterwards):
        'psd' - power spectral density
        'pow' - power spectrum
        """
        s = start + i * width / 2
        y = df[(df.index>=s) & (df.index<=s+width)]
        w = self.hann((y.index.to_series() - s) / width)
        y = self._lsdft(y * w)

        print('segment length: {}'.format(len(w)))
        if norm=='psd':
            return y * (width / np.sum(w**2)) ** .5
        elif norm=='pow':
            return y * len(w) ** .5 / np.sum(w)

    def welch(self, n, f=None, norm='psd'):
        if f is not None:
            self.set_freq(f)

        # lsdft is applied per-column
        def lsdft(c):
            y = c.dropna()
            start = y.index[0]
            T = y.index[-1] - start
            w = T / n
            win = partial(self._window, y, start, w, norm)
            return pd.concat([win(i) for i in range(2 * n - 1)], 1)

        return pd.Panel(dict([(name, lsdft(col)) for name, col in self._y.iteritems()]))

    def power(self, f=None, n=0, norm='psd', alpha=None):
        if n==0:
            p = 2 * np.abs(self.raw(f, norm))**2
        else:
            p = 2 * (self.welch(n, f, norm).abs()**2).mean(2)
            if alpha is not None:
                ne = 2 * self.neff(n)
                a = alpha / 2
                return p, ne / chi2.ppf([a, 1-a], ne)
            else: return p

    def amplitude(self, f=None, n=0):
        "Returns peak amplitude (= 1/2 peak-to-peak)"
        if n==0:
            return 2 * np.abs(self.raw(f, 'pow'))
        else:
            return 2 * (self.welch(n, f, 'pow').abs()**2).mean(2) ** .5

    def csd(self, n=0, f=None):
        if n==0:
            G = self.raw(f)
            return (G.iloc[:,0] * np.conj(G.iloc[:,1]))
        else:
            G = self.welch(n, f)
            return (G.iloc[0] * np.conj(G.iloc[1])).mean(1)

    @staticmethod
    def neff(n, ovc=0.167):
        return (2 * n - 1) / (1 + ovc ** 2 * (2 - 2 / n))

    def coherence(self, n=8, f=None):
        G = self.welch(n, f)
        XY = (G.iloc[0] * np.conj(G.iloc[1])).mean(1)
        return np.abs(XY)**2 / (np.mean(G.iloc[0].abs()**2, 1) * np.mean(G.iloc[1].abs()**2, 1))



if __name__=="__main__":
    from astropy.stats import LombScargle
    N = 256
    C = 16
    s = 1
    t = np.linspace(0, C, N)
    y = np.sin(2* np.pi * t) + np.random.rand(N) * s

    f = np.fft.fftfreq(N, t[1]-t[0])
    i = (f>0)
    f = f[i]
    P = abs(np.fft.fft(y)[i]) ** 2 / N

    L = LombScargle(t, y).power(f, normalization='psd')
    ls = LS(pd.DataFrame(y, index=t))
