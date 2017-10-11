#!/usr/bin/env python
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import tensorflow as tf
import edward as ed

# N - number of examples ('time')
# D - dimension of example ('space')
# K - number of principal components

def detPCA(X, K=5):
    w, v = np.linalg.eigh(np.cov(X)) # eigh for symmetric / hermitian matrices
    z = v[:,-K:].T.dot(X) # K x N - principal components
    y = v[:,-K:].dot(z)   # K x N - reconstruction
    print(np.abs(X-y).sum())

def probPCA(X, K=5, n_iter=500):
    w = ed.models.Normal(tf.zeros((5, K)), tf.ones((5, K)))
    z = ed.models.Normal(tf.zeros((K, 5000)), tf.ones((K, 5000)))
    x = ed.models.Normal(tf.matmul(w, z), tf.ones((5, 5000)))
    qw = ed.models.Normal(tf.Variable(tf.random_normal(w.shape)),
                          tf.nn.softplus(tf.Variable(tf.random_normal(w.shape))))
    qz = ed.models.Normal(tf.Variable(tf.random_normal(z.shape)),
                          tf.nn.softplus(tf.Variable(tf.random_normal(z.shape))))

    if1 = ed.KLqp({w: qw, z:qz}, data={x: X})
    out = if1.run(n_iter=n_iter)
    s = ed.get_session()
    y1 = s.run(qw.mean()).dot(s.run(qz.mean()))
    print(np.abs(X-y1).sum())

    y = tf.matmul(qw, qz)
    print(np.abs(X - s.run(y)).sum())

    y = tf.matmul(qw.mean(), qz.mean())
    print(np.abs(X - s.run(y)).sum())


if __name__=='__main__':
    w = np.random.normal(0, 1, (5, 5))
    z = np.random.normal(0, 1, (5, 5000))
    X = w.dot(z) + np.random.normal(0, 1, (5, 5000))

    detPCA(X)
