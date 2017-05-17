#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.signal as sig


def welch(t, x, n):
    w = sig.hann(n, sym=False)
    d = n // 2
    m = np.mean([LombScargle(t[j*d:j*d+n], x[j*d:j*d+n] * w).power(1, normalization='psd')
        for j in range(len(x) // d - 1)])
    return 2 * np.sqrt(m / n)

class LS(object):
    def __init__(self, df, tshift=0):
        t = np.array(df.index, dtype='datetime64[m]').astype(int)
        base_f = np.timedelta64(1,'Y').astype('timedelta64[m]').astype(int)
        self._t = np.r_['1,2', t].T / base_f
        self._y = df
        self._tf = tshift

    def tau(self, w):
        w2 = 2 * w
        wt = self._t * w
        tau = 1 / w2 * np.arctan(np.sum(np.sin(wt), 0) / np.sum(np.cos(wt), 0))
        tp = self._t - tau
        self._F0 = 1/(2**.5) * np.exp(-1j * w * (self._tf - tau))
        self._wtp = tp * w

    def Xk(self, f):
        self.tau(2 * np.pi * f)
        def dft(c):
            j = c.notnull()
            y = c[j]
            wtp = self._wtp[j,:]
            A = np.sum(np.cos(wtp)**2, 0) ** (-0.5)
            B = np.sum(np.sin(wtp)**2, 0) ** (-0.5)
            return pd.Series(self._F0 *
                             (A * y.dot(np.cos(wtp)) + 1j * B * y.dot(np.sin(wtp))),
                              index = f
                          )
        return self._y.apply(dft, 0)

    def coherence(self, f):
        X = self.Xk(f)
        return np.abs(X.iloc[:,0] * np.conj(X.iloc[:,1])) ** 2 / \
            ((np.abs(X.iloc[:,0]) ** 2) * (np.abs(X.iloc[:,1]) ** 2))

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
