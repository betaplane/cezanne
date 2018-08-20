"""
CEAZAMet station data comparison tools
--------------------------------------
"""
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

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

class Comp2(object):
    def __init__(self, a, b):
        # just checking that data_pc is all 0 or 100
        assert(not any([(lambda c: np.any((c!=0) & (c!=100)))(c.xs('data_pc', 1, 'aggr')) for c in [a, b]]))

        self.a, self.b = [c.drop('data_pc', 1, 'aggr') for c in [a, b]]
        x = (lambda d: (d == d) & (d != 0))(self.a - self.b)
        # locations where any of the non-data_pc aggr levels are not equal to old data
        self.x = reduce(np.add, [x.xs(a, 1, 'aggr') for a in x.columns.get_level_values('aggr').unique()])

        # stations where not only the last timestamp of old data differs from new data
        z = self.x.apply(lambda c: c.index[c].max() - c.index[c].min(), 0)
        self.s = z[(z == z) & (z > pd.Timedelta(0))]
        print(self.s)

    def plot(self, stations=None, dt=pd.Timedelta(1, 'D')):
        sta = self.s.index if stations is None else self.s.index[stations]
        fig, axs = plt.subplots(len(sta), 1)
        for i, ax in zip(sta, axs):
            x, y, z = [j.xs(i[2], 1, 'sensor_code') for j in [self.a, self.b, self.x]]
            d = z.index[z.values.flatten()]
            a, b = d.min() - dt, d.max() + dt
            for idx, c in y.iteritems():
                pl = ax.plot(c.loc[a: b], label=idx[-1])[0]
                ax.plot(x.xs(idx[-1], 1, 'aggr').loc[d], 'x', color=pl.get_color())
                ax.vlines(d, *ax.get_ylim(), color='r', label='_')
        plt.legend()
