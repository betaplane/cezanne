import pandas as pd
import numpy as np

class Lanczos(object):
    """Class to apply a (lowpass) Lanczos filter over a :class:`~pandas.DataFrame` by means of the :meth:`~pandas.DataFrame.rolling` method. Weights are precomputed, and the boundaries are treated by truncating the weights. The class needs to be initialized first and supplies the correct keyword arguments for the :meth:`~pandas.DataFrame.rolling call (as attribute :attr:`.roll`)::

        df = pandas.DataFrame(...)
        W = Lanczos(df, '10D', 3)
        filtered = df.rolling(**W.roll).apply(W)

    :param df: DataFrame to which the filtering is to be applied
    :type df: :class:`~pandas.DataFrame`
    :param period: `pandas offset alias <https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`_ giving the cutoff period.
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
