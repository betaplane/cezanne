#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AffinityPropagation


def distance_weighted_ave(x, dist):
    dm = dist.mean()
    i = dm[dm < 5e5].index
    dist = dist.loc[i, i]
    w = np.exp(- (dist / dist.mean().mean()) ** 2).replace(1, 0)
    d, w = x.align(w, axis=1, level='station')
    w.fillna(0, inplace=True)
    n = d.std()
    d = (d - d.mean()) / n
    v = d.fillna(0).dot(w.T) / d.notnull().astype(float).dot(w.T)
    return v.mul(n, axis=1, level='station')

class Blocks(object):
    def __init__(self, cluster_obj, **kwargs):
        """
        Example usage:
        B = Blocks(DecisionTreeClassifier, min_samples_leaf = 1000)
        B = Blocks(AffinityPropagation)
        B = Blocks(AffinityPropagation, preference = -4)

        B.compute(x)
        B.regress(y)
        """
        self.cl = cluster_obj(**kwargs)

    @staticmethod
    def indexer(notnull):
        d = {}
        for i, r in enumerate(notnull):
            try:
                d[tuple(r)].append(i)
            except KeyError:
                d[tuple(r)] = [i]
        return d

    def compute(self, x):
        self.X = x
        n = x.notnull().astype(int)
        if isinstance(self.cl, DecisionTreeClassifier):
            t = np.array(x.index, dtype='datetime64[m]', ndmin=2).astype(float).T
            self.dict = self.indexer(n.apply(lambda c: self.cl.fit(t, c).predict(t), 0).values)
        else:
            k, v = zip(*self.indexer(n.values).items())
            self.z = zip(v, self.cl.fit_predict(k))
            self.dict = {}
            for i, j in self.z:
                J = tuple(self.cl.cluster_centers_[j, :])
                try:
                    self.dict[J].extend(i)
                except KeyError:
                    self.dict[J] = i
        self.blocks = [x.iloc[i, np.array(c, dtype=bool)].dropna(1, 'all')
                    for c, i in self.dict.items()]

    def check(self):
        """Returns a pandas.DataFrame of same shape as original data, with clustered 1-0 arrangement corresponding to original missing value matrix."""
        b = pd.DataFrame(columns = self.X.columns)
        for k, v in self.dict.items():
            J = np.array(k, ndmin=2).repeat(len(v), 0)
            b = b.append(pd.DataFrame(J, columns=self.X.columns, index=self.X.index[v]))
        return b

    def regression(self, x, y):
        x0 = x.fillna(0)
        x0[1] = 1
        b = np.linalg.lstsq(x0, y.loc[x.index].values.flatten())[0]
        self.b.append(pd.Series(b[:-1], index=x.columns))
        self.c.append(len(x))
        return x0.dot(b.reshape((-1, 1)))

    def regress(self, y, x=None, blocks=True):
        y = y.dropna()
        if x is not None:
            self.compute(x.loc[y.index])
        self.b = []
        self.c = []
        if blocks:
            r = pd.concat([self.regression(b, y) for b in self.blocks]).sort_index()
        else:
            r = self.regression(pd.concat(self.blocks, 1).sort_index(), y)
        r = pd.DataFrame(r)
        if isinstance(y, pd.Series):
            r.columns = pd.MultiIndex.from_tuples([y.name], names = x.columns.names)
        else:
            r.columns = y.columns
        return r


if __name__ == "__main__":
    import binning, data

    D = data.Data()
    D.open('r','s_raw.h5')
    X = binning.bin(D.r).xs('avg', 1, 'aggr')
    X = X - X.mean()

    y = X.xs('3', 1, 'station', False).iloc[:,0].dropna()
    x = X.drop(y.name, 1).loc[y.index]

    # b = tree_blocks(x)
    # a = block_predictors(x, b)
    # a0 = pd.concat(a, 1).fillna(0)
    # r1 = np.linalg.lstsq(a0, y)
    # r2 = pd.concat([regression(c, y) for c in a], 0).sort_index()
    # r3 = affinity_regression(x, y)
