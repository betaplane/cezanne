"""
Tests
-----

"""

import pandas as pd
import numpy as np
from datetime import datetime
from importlib import import_module
import pca.core as core
import os


class Data(object):
    """Data producer for test cases. Saves original principal components :attr:`Z` and loadings :attr:`W` for loss computations.

    """

    real_data = {
        'ta_c': [
            ['3','4','5','8','9'],
            ['8', 'INILLA', 'LCAR', 'QS', 'TLH']
        ]
    }

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

    def real(self, config=None, **kwargs):
        k, v = kwargs.popitem()
        if config is None:
            config = import_module('cezar')
        t = pd.read_hdf(config.stations['data'], k).xs('prom', 1, 'aggr')[self.real_data[k][v]]
        sta = t.columns.get_level_values('station')
        if len(sta.get_duplicates()) > 0:
            t.columns = t.columns.get_level_values('sensor_code')
        else:
            t.columns = sta

        x = t.resample('D').mean().dropna(0, 'any')
        # this is a fraction of the data without any missing values
        # self.x = x[(x.index >= pd.Timestamp('2013')) & (x.index < pd.Timestamp('2017'))]

        i, = np.where(np.diff(x.index).astype('timedelta64[D]').astype(int)>1)
        if len(i) > 1:
            j = np.diff(i).argmax()
            self.x = x.iloc[i[j]+1: i[j+1]+1].T
        elif len(i) == 0:
            self.x = x.T
        else:
            raise Exception('not implemented')

        self.x1 = np.ma.masked_invalid(self.x)
        self.mask = pd.DataFrame(self.x.notnull(), index=self.x.index, columns=self.x.columns)
        return self

    def missing(self, frac, blocks=0):
        mask = np.ones(self.x.shape).flatten()
        N = int(round(frac * len(mask)))
        if blocks == 0:
            mask[np.random.choice(len(mask), N, replace=False)] = 0
        else:
            bl = N / blocks
            idx = np.arange(len(mask))
            n = N
            while n > 0:
                s = min(n, np.random.poisson(bl, 1).item()) # block lengths
                j = np.random.choice(idx[mask.astype(bool)][:-s], 1).item()
                mask[j: j + s] = 0
                n = int(mask.sum() - len(mask) + N)
        mask = np.reshape(mask, self.x.shape, {0: 'F', 1:'C'}[np.argmax(self.x.shape)])
        self.mask = pd.DataFrame(mask, index=self.x.index, columns=self.x.columns)
        x1 = self.x * self.mask.replace(0, np.nan)
        self.x1 = np.ma.masked_invalid(x1)
        return self

    @property
    def missing_fraction(self):
        return self.x1.mask.sum() / np.prod(self.x1.shape)


class Test(object):
    """Test runner class for the :mod:`~.pca.core` submodule. Method :meth:`case` can be used as a decorator to produce the necessary :class:`DataFrames<pandas.DataFrame>` for PCA configuration.

    :Arguments:
        * **file_name** - Name of the :class:`~pandas.HDFStore` file which holds or will hold the experiment specifications and results.
        * **test_name** - Top-level key under which the DataFrames specifying the experiments reside. There will be three nodes under the ``test_name``::
                * **args** - the DataFrame specifying the individual experiments, on row per experiment
                * **config** - a DataFrame containing the configurations referred to in the **args** DataFrame
                * **results** - an appendable DataFrame which will be populated while the experiments are run

    :Keyword Arguments:
        * **data** - The name of a file which saves the :class:`Data` instances used for the experiments, for repeatability and restart in case of a crash.
        * **plot** - If ``True``, will only read in the file given in ``file_name`` so that it can be used by the :meth:`plot` method.

    .. py:decoratormethod:: case(file_name)

        Decorator to create a :class:`~pandas.HDFStore` which contains all the specifications to run a series of experiments with the :mod:`~.pca.core` module. To use it decorate a function which returns either of:
            1. **One DataFrame** - this DataFrame describes the arguments to be passed to the :class:`~pca.core.PCA` constructor (e.g. ``W``, ``Z``, ``seed`` or annotation keywords).
            2. **(A tuple of) two DataFrames** - The first of the two is the same as in the first option; the second one contains hierarchically concatenated (along index) configuration DataFrames which need to be referred to by integer numbers in a column labelled 'config' in the first (argument) DataFrame. In other words, it contains one or more DataFrames representing modified versions of the default configuration obtained by a call to :meth:`~.pca.core.probPCA.configure`, concatenated bt a call to :func:`pandas.concat` with ``axis=0`` and ``keys`` a list of unique integers corresponding to the values of the 'config' column in the argument DataFrame.

        The argument ``file_name`` is the name of the :class:`~pandas.HDFStore` file in which the argument / configuration DataFrames are to be saved. The same file will be used to append the results from the experiments (see :meth:`~.pca.core.PCA.critique`). The **name of the function** is used as the first level in the hierarchical HDF5 file to denote a given experiment / test.
    """

    def _store_get(self, key):
        return self.store.get(key) if key in self.store else None

    def __init__(self, file_name, test_name, data='data.pkl', plot=False):
        import joblib
        self.table = test_name + '/results'
        with pd.HDFStore(file_name) as self.store:
            self.args = self._store_get(test_name + '/args')
            self.config = self._store_get(test_name + '/config')
            self.results = self._store_get(self.table)
        if not plot:
            self.keys = self.args.columns.difference({'data', 'seed', 'config', 'done'})
            if (self.results is not None) or os.path.isfile(data):
                self.data = joblib.load(data)
            if self.results is not None:
                self.row = self.results.index[-1] + 1
            else:
                self.data = {i :Data().toy() for i in self.args.get('data').unique()}
                joblib.dump(self.data, data)
                self.row = 0

    def run(self, n_iter=20000):
        if self.row > self.args.index[-1]:
            return -9
        print('row {}\n'.format(self.row))
        t = self.args.loc[self.row]
        d = self.data[t.get('data')]

        kwargs = t[self.keys].to_dict()
        conv = kwargs.pop('convergence_test', 100)
        config = None

        if t.get('config') is not None:
            config = self.config.loc[t.get('config')]

        # avoid creating new instances if unnecessary
        try:
            new_instance = (t.get('seed') != self.pca.seed) or np.all(config != self.pca.config)
        except TypeError: # config is None
            new_instance = (config is None) and np.all(self.pca.config != self.pca.configure())
        except AttributeError: # self.pca doesn't exist
            new_instance = True
        if new_instance:
            self.pca = core.probPCA(d.x1.shape, seed=t.get('seed'), config=config, **kwargs)
            print('new instance created\n')

        self.pca.run(d.x1, n_iter, convergence_test=conv)
        self.pca.critique(d, file_name=self.store.filename, table_name=self.table, row=self.row)
        self.row += 1
        return 0

    def unique(self):
        return self.results.drop(
            ['x', 'Z', 'W', 'n_iter', 'mu', 'tau', 'loss', 'data_loss', 'logs'], 1
            ).apply(lambda x: x.unique().size)

    def subplot(self, ax, data, column, xaxis, colors, offset=0, **kwargs):
        col, lab = list(colors.items())[0]
        label = kwargs.pop('label', lab[0])
        color = kwargs.pop('color', None)

        level, xlab = list(xaxis.items())[0]
        xlabels = kwargs.pop('xlabels', xlab)

        # this makes boolean lists for each kwarg and .all() is True only if all conditions are true
        # keys are column names, values are lists against which the column values are checked via a 'isin' query
        combined = [i for j in [xaxis, colors, kwargs] for i in j.items()]
        y = data[pd.concat([data[k].isin(v) for k, v in combined], 1).all(1)]

        # this groups the conditions
        y = y.groupby([str(k) for k, v in combined])[column]
        y = pd.concat((y.mean(), y.std()), 1, keys=['mean_', 'std_'])

        ymean = [y['mean_'].xs(i, 0, level).item() for i in xlab]
        ystd = [y['std_'].xs(i, 0, level).item() for i in xlab]

        x0 = np.arange(len(y))
        x = x0 + offset
        p = ax.errorbar(x, ymean, yerr=ystd, fmt='s', color=color, capsize=5, label=label)
        ax.set_xticks(x0)
        ax.set_xticklabels(xlabels)
        ax.set_title(column)

    def plot(self, xaxis, colors, **kwargs):
        """Produce a plot that splits the results of a :mod:`~.pca.core` experiment along two dimensions: the *xaxis* and differently *colored* plots. Plots the mean as a square and the standard deviation of the experiments falling into one particular group as whiskers. Shows 8 plots corresponding to attributes of the :class:`~.pca.core.PCA` subclasses (`x`, `Z`, `W`, `n_iter`, `mu`, `tau`, `loss`, `data_loss`).

        :param xaxis: Column of the ``results`` :class:`~pandas.DataFrame` whose values give rise to the groups displayed along the xaxis of the plots. If a :obj:`dict` is given, it should have *one* key: The column name by which to :meth:`~pandas.DataFrame.groupby` the results, and a :obj:`list` as the value, containing all the values of that columns which should be included. Alternatively, give only the column's name as a :obj:`str` to include all occuring values.
        :type xaxis: :obj:`dict` or :obj:`str`

        with the columns name as *one* key and the values to be included as a :obj:`list`, or a :obj:`str` if all occuring values should be included. (If a specific order of the items is desired, use the full dictionary specification).
        :param colors: Column of the ``results`` :class:`~pandas.DataFrame` whose values give rise to the groups displayed as separately colored plots with the same xaxes. The same rules apply as for `xaxis`.
        :type colors: :obj:`dict` or :obj:`str`
        :returns: The axes of the subplots, e.g. for placing a legend (labels are added automatically corresponding to the **colors** argument).
        :rtype: :class:`~numpy.ndarray` of :class:`~matplotlib.axes.Axes` instances

        :Keyword Arguments:
            Additional arguments specified as ``key=list`` will be used to restrict the results to the occurences of the values in ``list`` in the column named ``key``.

            The following keywords are reserved:

            * **results** - Pass in a ``results`` :class:`~pandas.DataFrame` other than the one read from the :class:`~pandas.HDFStore` given in the class initialization.
            * **figsize**, **hspace** and **wspace** are passed to the corresponding :mod:`matplotlib` calls.

        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=kwargs.pop('figsize', (12, 6)))
        fig.subplots_adjust(hspace=kwargs.pop('hspace', 0.3), wspace=kwargs.pop('wspace', .3))

        results = kwargs.pop('results', self.results)
        if isinstance(xaxis, str):
            xaxis = {xaxis: results[xaxis].unique()}
        if isinstance(colors, str):
            col = colors
            values = results[col].unique()
        else:
            col, values = colors.popitem()

        # below are just plot arrangements
        n = len(values)
        max = .25 * (1 - np.exp((1-n)/2))
        offs = np.linspace(-max, max, n)
        m = len(list(xaxis.values())[0])
        xlims = [-m/4, (m-1)+m/4]

        for k, x in enumerate(['x', 'Z', 'W', 'n_iter', 'mu', 'tau', 'loss', 'data_loss']):
            i = k // 4
            j = k % 4
            for l, v in enumerate(values):
                self.subplot(axs[i, j], results, x, xaxis, dict([(col, [v])]), offs[l], **kwargs)
                axs[i, j].set_xlim(xlims)
        return axs

    @staticmethod
    def case(file_name):
        def wrap(func):
            def wrapped_func(*args, **kwargs):
                out = func(*args, **kwargs)
                arg_name = os.path.join(func.__name__, 'args')
                with pd.HDFStore(file_name) as store:
                    if isinstance(out, pd.DataFrame):
                        store[arg_name] = out
                    else:
                        # the 'config' values are sorted so that graphs need not be reconstructed unnecessarily
                        # (the graph only needs to be reconstructed if 'config' or the data shape changes)
                        store[arg_name] = out[0]
                        store[os.path.join(func.__name__, 'config')] = out[1]
                args = store[arg_name]
                s = [i for i in ['config', 'seed'] if i in args.columns]
                if len(s) > 0:
                    store[arg_name] = args.sort_values(s).reset_index(drop=True)
            return wrapped_func
        return wrap


@Test.case('tests.h5')
def test0():
    args = pd.DataFrame({'data': [0, 1], 'config':[None, None], 'tttessst':[9, 10]})
    return args

@Test.case('convergence.h5')
def convergence(n_data=10, n_seed=10):
    import tensorboard as tf
    tests = pd.DataFrame()

    for i in range(n_data):
        for s in range(n_seed):
            for conv in ['data_loss', 'elbo']:

                tests = tests.append({'data': i, 'seed': s, 'convergence_test': conv,
                                      'config': 'none', 'covariance': 'none', 'W': 'none', 'Z': 'none'}
                                     , ignore_index=True)
                tests = tests.append({'data': i, 'seed': s, 'convergence_test': conv,
                                      'config': 0, 'covariance': 'full', 'W': 'all', 'Z': 'all'}
                                     , ignore_index=True)
    c = core.probPCA.configure()
    c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
    c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]
    conf = pd.concat((c,), 0, keys=[0])
    return tests, conf

@Test.case('covariance.h5')
def mu_tau(n_seed=10):
    import tensorboard as tf
    c = []
    t = pd.DataFrame()
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
                            c.append(core.probPCA.configure())
                            if mu=='full':
                                c[-1].loc[('prior', 'mu', 'loc'), :] = mu_loc
                                c[-1].loc[('prior', 'mu', 'scale'), :] = mu_scale
                            if tau=='full':
                                c[-1].loc[('prior', 'tau', 'loc'), :] = tau_loc
                                c[-1].loc[('prior', 'tau', 'scale'), :] = tau_scale

                            for s in range(n_seed):
                                t = t.append({'seed':s, 'mu':mu, 'tau':tau, 'config':len(c)-1, 'i':i, 'j':j, 'k':k, 'l':l, 'm':m, 'n':n, 'data':0}, ignore_index=True)

    config = pd.concat(c, 0, keys=range(len(c)))
    return t, config

@Test.case('experiments_copy.h5')
def covariance(n_data=10, n_seed=10):
    import tensorboard as tf
    t = pd.DataFrame()
    c = []
    for j, conf in enumerate([
            [False, tf.ones_initializer],
            [True, tf.ones_initializer],
            [True, tf.random_normal_initializer],
    ]):
        c.append(core.probPCA.configure())
        c[-1].loc[('prior', slice(None), 'scale'), :] = conf
        for i, kv in enumerate([
                {},
                {'W': 'prior'},
                {'Z': 'prior'},
                {'W': 'prior', 'Z': 'prior'},
                {'W': 'all'},
                {'Z': 'all'},
                {'W': 'all', 'Z': 'all'}
        ]):
            for s in range(n_seed):
                for d in range(n_data):
                    kv.update({'data': d, 'config': j, 'covariance': i, 'initialization': j, 'seed': s})
                    t = t.append(kv, ignore_index=True)
    config = pd.concat(c, 0, keys=range(len(c)))
    return t, config

# if __name__=='__main__':
#     out = 0

#     test = Test('experiments_copy.h5', 'covariance', data='covariance.pkl')
#     while out == 0:
#         out = test.run()
