import pandas as pd
import numpy as np
from functools import singledispatch
from importlib import import_module

@singledispatch
def LocalRegression(x, y, xi=None, kernel='RBF', length_scale=1, deg=1):
    """Local regression of arbitrary order and at arbitrary x locations. Similar to the R 'loess' function (I think). Instead of the arguments ``x``, ``y`` and ``xi``, a single :class:`pandas.DataFrame` can be subsituted, in which case ``x`` and ``y`` are taken to be its :attr:`~pandas.DataFrame.index` and :attr:`~pandas.DataFrame.values` after applying :meth:`~pandas.DataFrame.dropna`, while ``xi`` is taken to be the index before dropping the NaNs. If a :class:`~pandas.DataFrame` is passed as input, the output will also be one.

    :param x: independent variable
    :type x: numeric type :class:`~numpy.ndarray`
    :param y: dependent variable
    :type y: numeric type :class:`~numpy.ndarray`
    :param xi: locations at which to evaluate the regression (defaults to ``x``)
    :type xi: numeric type :class:`~numpy.ndarray`
    :param kernel: kernel type to use, amount those found in `sklearn.gaussian_process.kernels <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process>`_
    :mod:`sklearn.gaussian_process.kernels`
    :type kernel: :obj:`str`
    :param length_scale: argument ``length_scale`` for the instantiation of the kernel
    :type deg: numeric
    :param deg: degree of the local regression (linear: 1, quadratic: 2, etc)
    :type deg: :obj:`int`
    :returns: regression result evaluated at ``x`` or ``xi``
    :rtype: :class:`~numpy.ndarray` or :class:`~pandas.DataFrame`

    """
    kernels = import_module('sklearn.gaussian_process.kernels')
    linalg = import_module('scipy.linalg')
    k = getattr(kernels, kernel)(length_scale)

    H = np.array([x.flatten()**i for i in range(deg*2+1)])[linalg.hankel(range((deg+1)*2))[:deg+1, :deg+1]]
    if xi is None:
        K = k(x.reshape((-1, 1)))
        b = H[0, :, :]
    else:
        K = k(x.reshape((-1, 1)), xi.reshape((-1, 1)))
        b = np.array([xi.flatten()**i for i in range(deg+1)])
    Inv = np.linalg.inv(np.einsum('ijk,kl', H, K).T)
    k = np.einsum('ij,jil->jl', b, Inv)
    l = np.einsum('ij,jk,j', H[0, :, :], K, y.flatten())
    return np.einsum('ij,ji->i', k, l)

@LocalRegression.register(pd.DataFrame)
def _(df, datetime_granularity='m', **kwargs):
    # scaling is necessary for higher-order regressions due to numerical instability with high powers - if there are any, improve the 's' lambda
    s = lambda x: x-x[0] # at this point only sets starting point to 0

    dtype = 'datetime64[{}]'.format(datetime_granularity)
    y = df.dropna()
    xi = None
    if isinstance(df.index, pd.DatetimeIndex):
        x = s(np.array(y.index, dtype=dtype).astype(float))
        if y.shape[0] < df.shape[0]:
            xi = s(np.array(df.index, dtype=dtype).astype(float))
    else:
        x = s(y.index)
        if y.shape[0] < df.shape[0]:
            xi = s(df.index)
    return pd.DataFrame(LocalRegression(x, y.values, xi, **kwargs), index=df.index, columns=df.columns)


class Lanczos(object):
    """Class to apply a (lowpass) Lanczos filter over a :class:`~pandas.DataFrame` by means of the :meth:`~pandas.DataFrame.rolling` method. Weights are precomputed, and the boundaries are treated by truncating the weights. The class needs to be initialized first and supplies the correct keyword arguments for the :meth:`~pandas.DataFrame.rolling call (as attribute :attr:`.roll`)::

        df = pandas.DataFrame(...)
        W = Lanczos(df, '10D', 3)
        filtered = df.rolling(**W.roll).apply(W)

    :param df: DataFrame to which the filtering is to be applied (needs to have an index with :attr:`~pandas.DatetimeIndex.freq` - apply :meth:`~pandas.DataFrame.resample` first if that isn't the case).
    :type df: :class:`~pandas.DataFrame`
    :param period: :class:`pandas.Timedelta` :obj:`str` giving the cutoff period.
    :type period: :obj:`str`
    :param a: integer 'order' of the filter (the higher, the more filter terms)
    :type a: :obj:`int`
    """
    def __init__(self, df, period, a=3):
        N = int(pd.Timedelta(period) / df.index.freq * a + 1)
        self.s = df.shape[0]
        self.n1 = N // 2
        self.n2 = self.s - self.n1
        x = np.linspace(-a, a, N)
        self.w = np.sinc(x) * np.sinc(x / a)
        self.roll = {'window': N, 'center': True, 'min_periods': 0}
        self.i = 0 # running index of row location

    def __call__(self, x):
        if self.i < self.n1: # if at beginning, take the end points of the filter
            w = self.w[-len(x):]
        elif self.i >= self.n2: # if at end, take the beginning of the filter
            w = self.w[:len(x)]
        else:
            w = self.w
        self.i = (self.i + 1) % self.s
        return np.nansum(x * w) / w[np.isfinite(x)].sum()

    @classmethod
    def filter(cls, df, period, a=3):
        fil = cls(df, period, a)
        return df.rolling(**fil.roll).apply(fil)
