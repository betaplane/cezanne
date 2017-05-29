#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.linalg import solve
import helpers as hh


class reg_base(object):
    def __init__(self, dz, dist, s=20000, c=50000):
        self._dz = dz.dropna()
        i = self._dz.index
        w = dist.loc[i, i]
        w[w > c] = np.nan
        self._w = np.exp(-(w / s) ** 2)

    def plot(self, X, loc):
        fig = plt.figure()
        w = self._w[loc]
        w = w[w>0]
        i = w.index
        x = X[i].as_matrix().T.flatten()
        plt.scatter(np.repeat(self._dz[i], len(X)), x, c=np.repeat(w, len(X)))
        plt.colorbar()
        plt.scatter(np.repeat(self._dz[loc], len(X)), X[loc], color='r')
        xl = plt.gca().get_xlim()
        y = np.linspace(xl[0], xl[1], 100)
        plt.plot(y, self.p['icpt'][loc] + y * self.p['lr'][loc], '-')
        fig.show()

class OLS(reg_base):
    def __init__(self, *args, **kw):
        self._weights = kw.pop('weights', False)
        super(OLS, self).__init__(*args, **kw)

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
        x, y = self._dz.align(t.T, broadcast_axis=1)
        f = pd.Panel({'x':x, 'y':y}).to_frame()
        f[1] = 1
        self.p = self._w.apply(self._mspace, data=f).T

    @staticmethod
    def _grouper(t, s=0, h=3):
        return (np.array(t, dtype='datetime64[h]').astype(int) - s) // 3

    def regress(self, X, mult=True):
        if mult:
            self._mtime(X)
        else:
            X.apply(self._time)
        return self.p['lr'] * 1000


class GLR(reg_base):
    def __init__(self, *args, **kw):
        super(GLR, self).__init__(*args, **kw)
        i = self._dz.index
        j = pd.MultiIndex.from_product((i, [0, 1]))
        self._i = dict([(k, pd.MultiIndex.from_product((i, [k]))) for k in [0, 1]])
        m = self._w.notnull()
        self._w.fillna(0, inplace=True)
        self.L = pd.DataFrame(
            np.kron(np.diag(self._w.sum(1)) - self._w, np.identity(2)),
            index = j, columns = j)

        # m = pd.DataFrame(np.identity(len(w)), index=i, columns=i)
        # [ 1]
        # [dz]
        self._m = pd.concat((m, m.mul(self._dz, 1)), 0)
        self._m.index = pd.MultiIndex.from_product(([0, 1], i)).swaplevel()

    def regress(self, X, lda):
        x = X[self._m.columns].T.notnull()
        # do identical operation on each timestep, then sum over times
        # [1]
        #    [dz**2]
        ad = np.sum((self._m ** 2).dot(x), 1)
        #      [dz]
        # [dz]
        bc = np.sum((self._m.xs(0, 0, 1) * self._m.xs(1, 0, 1)).dot(x), 1)
        self.A = self._diag(ad.xs(0, 0, 1), 0, 0) + self._diag(ad.xs(1, 0, 1), 1, 1)
        self.A = self.A + self._diag(bc, 0, 1) + self._diag(bc, 1, 0)
        self.C = np.sum(self._m.dot(X[self._m.columns].T.fillna(0)), 1)
        i = self.C.index
        p = solve(self.A.loc[i, i] + lda * self.L.loc[i, i], self.C)
        self.p = pd.DataFrame(p, index=i).unstack()
        self.p.columns = ['icpt', 'lr']
        return self.p['lr'] * 1000

    def _diag(self, ii, row, col):
        a = pd.DataFrame(np.diag(ii), index = self._i[row], columns = self._i[col])
        z = np.zeros(a.shape)
        b = pd.DataFrame(z, index = self._i[row], columns = self._i[not col])
        c = pd.DataFrame(z, index = self._i[not row], columns = self._i[col])
        d = pd.DataFrame(z, index = self._i[not row], columns = self._i[not col])
        return pd.concat((
            pd.concat((a, b), 1).sort_index(1),
            pd.concat((c, d), 1).sort_index(1)
        ), 0).sort_index(0)



if __name__=="__main__":
    D = pd.HDFStore('../../ceaza/data/station_data_new.h5')
    S = pd.HDFStore('../../ceaza/data/LinearLinear.h5')
    T = hh.stationize(D['ta_c'].xs('prom', 1, 'aggr').drop('10', 1, 'elev')) + 273.15
    Tm = S['T2']
    sta = D['sta']
    dist = D['dist']
    dz = S['z']['d03_op'] - sta['elev']
    dt = Tm['d03_0_00']-T
    D.close()
    S.close()
    X = dt.loc['2015-05-18']

    g = OLS(dz, dist)
    p = g.regress(X)
