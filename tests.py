"""
Tests
-----

"""

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import joblib
# import pca


class Data(object):
    """Data producer for test cases. Saves original principal components :attr:`Z` and loadings :attr:`W` for loss computations.

    """

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

def test2(file, n_iter=2000, n_seed=10):
    import pca
    d = data().toy()

    config = pca.probPCA.configure()
    for i, mu in [(1, 'full')]: #enumerate(['none', 'full']):
        for j, mu_loc in enumerate({
            'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
            'full': [
                [False, 'data_mean'], # prior mean set to data mean
                [True, 'data_mean']   # prior mean a hyperparamter
            ]
        }[mu][1:]):
            j = 1
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
                                p = pca.probPCA(d.x1, seed=s, mu=mu, tau=tau, config=c, i=i, j=j, k=k, l=l, m=m, n=n)
                                p.run(n_iter).critique(d)

                            with pd.HDFStore(file) as S:
                                S['exp2'] = p.losses.replace('None', np.nan)

def test3(file, n_iter=20000, n_seed=30):
    import pca
    d = data().toy()

    for s in range(n_seed):
        p = pca.probPCA(d.x1, seed=s, covariance=0).run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp3'] = p.losses.replace('None', np.nan)

    c = pca.probPCA.configure()
    c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
    c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]

    for s in range(n_seed):
        p = pca.probPCA(d.x1, seed=s, config=c, W='all', Z='all', covariance=1)
        p.run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp3'] = p.losses.replace('None', np.nan)

def test4(file, n_iter=20000, n_seed=30):
    import pca

    for s in range(n_seed):
        d = data().toy()
        c = pca.probpca.configure()

        p = pca.probpca(d.x1, covariance=0, config=c).run(n_iter).critique(d)

        c.loc[('prior', 'w', 'scale'), :] = [true, tf.random_normal_initializer]
        c.loc[('prior', 'z', 'scale'), :] = [true, tf.random_normal_initializer]

        p = pca.probpca(d.x1, config=c, w='all', z='all', covariance=1)
        p.run(n_iter).critique(d)

        with pd.hdfstore(file) as s:
            s['exp4'] = p.losses.replace('none', np.nan)

def test5(file='test5.h5', n_iter=20000, n_seed=10, n_data=10):
    import pca

    for i in range(n_data):
        d = data().toy()
        for s in range(n_seed):
            for conv in ['data', 'ed']:
                c = pca.probPCA.configure()

                p = pca.probPCA(d.x1, covariance=0, config=c, seed=s, conv=conv)
                p.run(n_iter, conv).critique(d)

                c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
                c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]

                p = pca.probPCA(d.x1, config=c, W='all', Z='all', seed=s, covariance=1, conv=conv)
                p.run(n_iter, conv).critique(d)

                with pd.HDFStore(file) as S:
                    s['test5'] = p.losses.replace('none', np.nan)


class Test(object):
    """Test runner class for the :mod:`~.pca.pca` submodule. Method :meth:`case` can be used as a decorator to produce the necessary :class:`DataFrames<pandas.DataFrame>` for PCA configuration."""

    def _store_get(self, key):
        return self.store.get(key) if key in self.store else None

    def __init__(self, file_name, test_name):
        self.results = test_name + '/results'
        with pd.HDFStore(file_name) as self.store:
            self.args = self._store_get(test_name + '/args')
            self.config = self._store_get(test_name + '/config')
            results = self._store_get(self.results)
        self.keys = self.args.columns.difference({'data', 'seed', 'config', 'done'})
        if results is not None:
            self.data = joblib.load('data.pkl')
            self.row = results.index[-1] + 1
        else:
            self.data = {i :Data().toy() for i in self.args.get('data').unique()}
            joblib.dump(self.data, 'data.pkl')
            self.row = 0

    def run(self, n_iter=20000):
        if self.row > self.args.index[-1]:
            return -9
        print('\nrow {}\n'.format(self.row))
        t = self.args.loc[self.row]
        d = self.data[t.get('data')]
        kwargs = t[self.keys].to_dict()
        config = None
        if t.get('config') is not None:
            config = self.config.loc[t.get('config')]
        self.p = pca.probPCA(d.x1, seed=t.get('seed'), config=config, **kwargs)
        self.p.run(n_iter).critique(d, file_name=self.store.filename, table_name=self.results, row=self.row)
        return 0

    @staticmethod
    def case(file_name):
        def wrap(func):
            def wrapped_func(*args, **kwargs):
                out = func(*args, **kwargs)
                with pd.HDFStore(file_name) as store:
                    if isinstance(out, pd.DataFrame):
                        store[func.__name__ + '/args'] = out
                    else:
                        store[func.__name__ + '/args'] = out[0]
                        store[func.__name__ + '/config'] = out[1]
            return wrapped_func
        return wrap


@Test.case('tests.h5')
def test():
    args = pd.DataFrame({'data': [0, 1], 'config':[None, None], 'tttessst':[9, 10]})
    return args

@Test.case('convergence.h5')
def data_loss_vs_elbo(n_data=10, n_seed=10):
    tests = pd.DataFrame()

    for i in range(n_data):
        for s in range(n_seed):
            for conv in ['data_loss', 'elbo']:

                tests = tests.append({'data': i, 'seed': s, 'convergence_test': conv,
                                      'config': None, 'covariance': 'none', 'W': 'none', 'Z': 'none'}
                                     , ignore_index=True)
                tests = tests.append({'data': i, 'seed': s, 'convergence_test': conv,
                                      'config': 0, 'covariance': 'full', 'W': 'all', 'Z': 'all'}
                                     , ignore_index=True)
    c = pca.probPCA.configure()
    c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
    c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]
    conf = pd.concat((c,), 0, keys=[0])
    return tests, conf

# if __name__=='__main__':
#     out = 0

#     while out == 0:
#         out = Test('convergence.h5', 'data_loss_vs_elbo').run()
