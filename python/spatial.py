#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def distance_weighted_ave(data, dist):
    w = np.exp(- (dist / dist.mean().mean()) ** 2).replace(1, 0)
    d, w = data.align(w, axis=1, level='station')
    w.fillna(0, inplace=True)
    d = (d - d.mean()) / d.std()
    return d.fillna(0).dot(w.T) / d.notnull().astype(float).dot(w.T)

def tree_blocks(data, min_samples_leaf=1000):
    t = np.array(data.index, dtype='datetime64[m]', ndmin=2).astype(float).T
    tr = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    def tree(c):
        return tr.fit(t, c.notnull().astype(float)).predict(t)
    return indexer(data.apply(tree, 0).values)

def block_indexes(data):
    return [data[data == np.array(i).reshape((1, -1))].dropna(0, 'any').index
            for i in set([tuple(r) for r in data.values])]

def block_predictors(data, block_dict):
    return [X.iloc[i, np.array(c, dtype=bool)].dropna(1, 'all')
            for c, i in block_dict.items()]

def regression(x, y):
    x0 = x.fillna(0)
    b = np.linalg.lstsq(x1, y.loc[x.index])[0]
    return x0.dot(b.reshape((-1, 1)))

def affinity_blocks(data, check=False):
    from sklearn.cluster import AffinityPropagation
    af = AffinityPropagation()
    n = data.notnull().astype(int)
    k, v = zip(*indexer(n.values).items())
    l = af.fit_predict(k)
    if check:
        b = pd.DataFrame(columns = n.columns)
        for i, j in zip(v, l):
            J = af.cluster_centers_[j:j+1, :].repeat(len(i), 0)
            b = b.append(pd.DataFrame(J, columns=n.columns, index=n.index[i]))
        return b
    else:
        d = {}
        for i, j in zip(v, l):
            J = tuple(af.cluster_centers_[j, :])
            try:
                d[J].extend(i)
            except KeyError:
                d[J] = i
        return d

def blocks_regression(X, y, blocks):
    def reg(c, i):
        x = X.iloc[i, np.array(c, dtype=bool)].dropna(1, 'all')
        xa = x.dropna(0, 'any')
        print(len(x), len(xa), sum(c))
        return regression(x, y)
    return pd.concat([reg(*ci) for ci in blocks.items()]).sort_index()

def indexer(notnull):
    d = {}
    for i, r in enumerate(notnull):
        try:
            d[tuple(r)].append(i)
        except KeyError:
            d[tuple(r)] = [i]
    return d

if __name__ == "__main__":
    import binning, data

    D = data.Data()
    D.open('r','s_raw.h5')
    X = binning.bin(D.r).xs('avg', 1, 'aggr')
    X = X - X.mean()

    y = X.xs('3', 1, 'station', False).iloc[:,0].dropna()
    x = X.drop(y.name, 1).loc[y.index]

    b = tree_blocks(x)
    a = block_predictors(x, b)
    a0 = pd.concat(a, 1).fillna(0)
    r1 = np.linalg.lstsq(a0, y)
    r2 = pd.concat([regression(c, y) for c in a], 0).sort_index()
    r3 = affinity_regression(x, y)
