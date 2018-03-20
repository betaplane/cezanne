import pandas as pd
import numpy as np

class Comp(object):
    """Comparator object to compare two CEAZA station data :class:`DataFrames<pandas.DataFrame>`.

    :Arguments:
        * **series** - the two :class:`DataFrames<pandas.DataFrame>` to be compared

    :Keyword Arguments:
        * **aggr** - which aggregation type (level ``aggr`` in columns :class:`~pandas.MultiIndex`) to compare (``prom``, ``min`` or ``max`` currently)

    """
    def __init__(self, *series, aggr='prom'):
        series = [s.xs(aggr, 1, 'aggr') for s in series]
        T = [s.dropna(0, 'all').index.max() for s in series]
        k = np.argmin(T)
        self.t = T[k]
        self.a, self.b = series[k].align(series[1-k])
        self.d = (self.a - self.b).abs().unstack().dropna().sort_values(ascending=False)
        last = self.d.xs(self.t, 0, 'ultima_lectura', drop_level=False)
        self.d.drop(last.index, inplace=True)
        self.iter = self.d[self.d > 0].iteritems()
        print('{} records at the end of the shorter time series, with errors between {} and {}, are being ignored.'.format(last.shape[0], last.min(), last.max()))

    def plot(self, delta='2D'):
        """Each call to :meth:`.plot` steps through the iterator (:attr:`.iter`) and plots the next (ordered from largest to smallest absolute value) mismatch in the data.

        :param delta: :class:`pandas.Timedelta` string denoting the time before and after each data mismatch which should be plotted
        :type delta: :obj:`str`

        """
        i, x = next(self.iter)
        pl = self.b[i[0]].plot()
        self.a[i[0]].plot(ax=pl.axes)
        dt = pd.Timedelta(delta)
        pl.axes.set_xlim(i[-1]-dt, i[-1]+dt)
