#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.linalg import solve
import helpers as hh
import matplotlib.pyplot as plt


class reg_base(object):
    def __init__(self, dz, dist):
        self._dz = dz.dropna()
        i = self._dz.index
        self._dist = dist.loc[i, i]

    def plot(self, X, loc):
        fig = plt.figure()
        w = self._w[loc]
        w = w[w>0]
        i = list(set(w.index).intersection(X.columns))
        x = X[i].as_matrix().T.flatten()
        plt.scatter(np.repeat(self._dz[i], len(X)), x, c=np.repeat(w[i], len(X)))
        plt.colorbar()
        plt.scatter(np.repeat(self._dz[loc], len(X)), X[loc], color='r')
        xl = plt.gca().get_xlim()
        y = np.linspace(xl[0], xl[1], 100)
        plt.plot(y, self.p['icpt'][loc] + y * self.p['lr'][loc], '-')
        fig.show()


# This is my initial ad-hoc implementation of spatio-temporally localized (over a neighborhood) linear regression
# for comparison purposes.

class OLS(reg_base):
    def __init__(self, *args, c=50000, s=20000, weights=False):
        super(OLS, self).__init__(*args)
        self._weights = weights
        self._w = np.exp( -(self._dist[self._dist < c] / s) ** 2 )

    def _space(self, v, data):
        try:
            y = data.mul(v, 0) if self._weights else data.loc[v.index[v.notnull()]]
            y.dropna(inplace=True)
        except:
            return pd.Series([np.nan, np.nan], index=['icpt', 'lr'])
        if len(y) < 5:
            return pd.Series([np.nan, np.nan], index=['icpt', 'lr'])
        x = y[[1,'x','y']].as_matrix()
        b = np.linalg.lstsq(x[:,:2], x[:,2])[0]
        return pd.Series(b, index=['icpt','lr'])

    def _time(self, t):
        a = self._dz.to_frame()
        a[1] = 1
        m = pd.concat((a, t), 1).dropna()
        self.p = self._w.apply(self._space, data=m).T

    def _mspace(self, v, data):
        try:
            y = data.mul(v, 0, level=0) if self._weights else data.loc[v.index[v.notnull()].tolist()]
            y = y.dropna()
        except:
            return pd.Series([np.nan, np.nan], index=['icpt', 'lr'])
        if len(y) < 5:
            return pd.Series([np.nan, np.nan], index=['icpt', 'lr'])
        x = y[[1,'x','y']].as_matrix()
        b = np.linalg.lstsq(x[:,:2], x[:,2])[0]
        return pd.Series(b, index=['icpt', 'lr'])

    def _mtime(self, t):
        c = t.columns
        x, y = self._dz[c].align(t.T, broadcast_axis=1)
        f = pd.Panel({'x':x, 'y':y}).to_frame()
        f[1] = 1
        self.p = self._w.loc[c,c].apply(self._mspace, data=f).T

    @staticmethod
    def _grouper(t, s=0, h=3):
        return (np.array(t, dtype='datetime64[h]').astype(int) - s) // 3

    def regress(self, X, mult=True):
        if mult:
            self._mtime(X)
        else:
            X.apply(self._time)
        return self.p['lr'] * 1000

# This is an implementation of a linear regression local in space and time (but over some neighborhood), with
# spatial smoothness of the regression coefficients enforced by regularization via a spatial graph Laplacian.
# Based on:
# Subbian, Karthik, and Arindam Banerjee. “Climate Multi-Model Regression Using Spatial Smoothing.”
# In SDM, 324–332. SIAM, 2013. http://epubs.siam.org/doi/abs/10.1137/1.9781611972832.36.

class GLR(reg_base):
    def __init__(self, *args, reg_mask, coeff_mask):
        super(GLR, self).__init__(*args)
        i = self._dz.index
        j = pd.MultiIndex.from_product((i, [0, 1]))

        # mask for the actual regression
        # m = (self._dist < c)
        m = reg_mask

        # the weights for the graph Laplacian
        # self._w = 1 - self._dist / self._dist.max().max()
        self._w = coeff_mask

        # no weighting would be identity here
        # m = pd.DataFrame(np.identity(len(w)), index=i, columns=i)

        self.L = pd.DataFrame(
            np.kron(np.diag(self._w.sum(1)) - self._w, np.identity(2)),
            index = j, columns = j)

        # This is problem-specifc: the construction of the design matrix from
        # constant and dz, which is constant in time. 
        # [ 1]
        # [dz]
        self._m = pd.concat((m, m.mul(self._dz, 1)), 0)
        self._m.index = pd.MultiIndex.from_product(([0, 1], i)).swaplevel()
        self._m.sort_index(inplace=True)

    def regress(self, X, lda):
        c = list(set(self._m.columns).intersection(X.columns))
        xt = X[c].T
        m = self._m.loc[c, c]
        x = xt.notnull()
        # from IPython.core.debugger import Tracer; Tracer()()
        # do identical operation on each timestep, then sum over times
        # [1]
        #    [dz**2]
        ad = np.sum((m ** 2).dot(x), 1)
        #      [dz]
        # [dz]
        bc = np.sum((m.xs(0, 0, 1) * m.xs(1, 0, 1)).dot(x), 1)
        self.A = self._diag(ad.xs(0, 0, 1), 0, 0) + self._diag(ad.xs(1, 0, 1), 1, 1)
        self.A = self.A + self._diag(bc, 0, 1) + self._diag(bc, 1, 0)
        self.C = np.sum(m.dot(xt.fillna(0)), 1) # possibly wrong
        i = self.C.index
        p = solve(self.A.loc[i, i] + lda * self.L.loc[i, i], self.C)
        self.p = pd.DataFrame(p, index=i).unstack()
        self.p.columns = ['icpt', 'lr']
        return self.p['lr'] * 1000

    def _diag(self, ii, row, col):
        ix = dict([(k, pd.MultiIndex.from_product((ii.index, [k]))) for k in [0, 1]])
        a = pd.DataFrame(np.diag(ii), index = ix[row], columns = ix[col])
        z = np.zeros(a.shape)
        b = pd.DataFrame(z, index = ix[row], columns = ix[not col])
        c = pd.DataFrame(z, index = ix[not row], columns = ix[col])
        d = pd.DataFrame(z, index = ix[not row], columns = ix[not col])
        return pd.concat((
            pd.concat((a, b), 1).sort_index(1),
            pd.concat((c, d), 1).sort_index(1)
        ), 0).sort_index(0)




if __name__=="__main__":
    D = pd.HDFStore('../../data/tables/station_data_new.h5')
    S = pd.HDFStore('../../data/tables/LinearLinear.h5')
    T = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, 'elev')) + 273.15
    Tm = S['T2n']
    sta = D['sta']
    dist = D['dist']
    dz = S['z']['d03_op'] - sta['elev']
    dt = Tm['d03_0_00']-T
    D.close()
    S.close()
    X = dt.loc['2015-05-18']

    g = OLS(dz, dist)
    p = g.regress(X)
