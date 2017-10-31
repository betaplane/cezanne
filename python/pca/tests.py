import pandas as pd
import numpy as np
from datetime import datetime


class data(object):
    def __init__(self):
        self.id = datetime.utcnow().strftime('dataf%Y%m%d%H%M%S%f')

    def toy(self, N=5000, D=5, K=5, tau=1):
        w = np.random.normal(0, 1, (D, K)) # weights
        z = np.random.normal(0, 1, (K, N)) # components
        self.mu = np.random.normal(0, 1, (D, 1)) # means
        x = w.dot(z)
        self.x = pd.DataFrame(x + self.mu)
        self.tau = tau
        self.x1 = np.ma.masked_invalid(self.x + np.random.normal(0, tau, (D, N)))

        # apply PCA to rotate W and Z for later error computation
        e, v = np.linalg.eigh(np.cov(x))
        self.W = v[:, np.argsort(e)[::-1][:K]]
        self.Z = self.W.T.dot(x).T
        return self

    def real(self):
        t = pd.read_hdf('../../data/CEAZAMet/station_data.h5', 'ta_c').xs('prom', 1, 'aggr')[['3','4','5','8','9']]
        t.columns = t.columns.get_level_values(0)
        x = t.resample('D').mean()
        self.mask = pd.DataFrame(x.notnull(), index=x.index, columns=x.columns)
        self.x1 = np.ma.masked_invalid(x)
        # this is a fraction of the data without any missing values
        self.x = x[(x.index >= pd.Timestamp('2013')) & (x.index < pd.Timestamp('2017'))]
        return self

    def missing(self, frac, blocks=0):
        mask = np.ones(self.x.shape).flatten()
        n = int(round(frac * len(mask)))
        if blocks == 0:
            mask[np.random.choice(len(mask), n, replace=False)] = 0
        else:
            s = np.random.poisson(n / blocks, blocks) # block lengths
            i = np.random.choice(len(mask) - round(s.mean()), blocks, replace=False) # block start indexes
            for j, t in zip(i, s):
                mask[j: j + t] = 0
        mask = np.reshape(mask, self.x.shape, {0: 'F', 1:'C'}[np.argmax(self.x.shape)])
        self.mask = pd.DataFrame(mask, index=self.x.index, columns=self.x.columns)
        x1 = self.x * self.mask.replace(0, np.nan)
        self.x1 = np.ma.masked_invalid(x1)
        return self

    @property
    def missing_fraction(self):
        return self.x1.mask.sum() / np.prod(self.x1.shape)
