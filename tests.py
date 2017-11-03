import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime


class data(object):
    def __init__(self):
        self.id = datetime.utcnow().strftime('data%Y%m%d%H%M%S%f')

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


def test1(file, n_iter=2000, n_seed=10):
    import pca
    d = data().toy()

    config = pca.probPCA.configure()
    for i, kv in enumerate([
            {'W': 'prior'},
            {'Z': 'prior'},
            {'W': 'prior', 'Z': 'prior'},
            {'W': 'all'},
            {'Z': 'all'},
            {'W': 'all', 'Z': 'all'}
    ]):
        for j, conf in enumerate([
                [False, tf.ones_initializer],
                [True, tf.ones_initializer],
                [True, tf.random_normal_initializer],
        ]):
            c = config.copy()
            for k, v in kv.items():
                c.loc[('prior', k, 'scale'), :] = conf
            for s in range(n_seed):
                p = pca.probPCA(d.x1, config=c, seed=s, covariance=i, initialization=j, **kv)
                p.run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp1'] = p.losses.replace('None', np.nan)

def test2(file, n_iter=2000):
    import pca
    d = data().toy()

    config = pca.probPCA.configure()
    for i, mu in enumerate(['none', 'full']):
        for j, mu_loc in enumerate({
            'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
            'full': [
                [False, 'data_mean'], # prior mean set to data mean
                [True, 'data_mean']   # prior mean a hyperparamter
            ]
        }[mu]):
            for k, mu_scale in enumerate({
                    'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                    'full': [
                        [False, tf.ones_initializer],
                        [True, tf.ones_initializer],
                        [True, tf.random_normal_initializer]
                    ]
            }[mu]):
                for l, tau in enumerate(['none', 'full']):
                    for m, tau_loc in enumerate({
                            'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                            'full': [
                                [False, tf.zeros_initializer],
                                [True, tf.zeros_initializer],
                                [True, tf.random_normal_initializer]
                            ]
                    }[tau]):
                        for n, tau_scale in enumerate({
                                'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                                'full': [
                                    [False, tf.ones_initializer],
                                    [True, tf.ones_initializer],
                                    [True, tf.random_normal_initializer]
                                ]
                        }[tau]):
                            c = config.copy()
                            c.loc[('prior', 'mu', 'loc'), :] = mu_loc
                            c.loc[('prior', 'mu', 'scale'), :] = mu_scale
                            c.loc[('prior', 'tau', 'loc'), :] = tau_loc
                            c.loc[('prior', 'tau', 'scale'), :] = tau_scale

                            for s in range(n_seed):
                                p = probPCA(d.x1, seed=s, mu=mu, tau=tau, config=x, i=i, j=j, k=k, l=l, m=m, n=n)
                                p.run(n_iter).critique(d)

                            with pd.HDFStore(file) as S:
                                S['exp2'] = p.losses.replace('None', np.nan)
