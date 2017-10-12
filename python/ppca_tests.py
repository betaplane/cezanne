#!/usr/bin/env python
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import tensorflow as tf
import edward as ed
import bayespy as bp

N = 5000  # number of examples ('time')
D = 5     # dimension of example ('space')
K = 5     # number of principal components

def detPCA(x_n, K=5):
    w, v = np.linalg.eigh(np.cov(x_n)) # eigh for symmetric / hermitian matrices
    z = v[:,-K:].T.dot(x_n)            # K x N - principal components
    x = v[:,-K:].dot(z)                # K x N - reconstruction
    return x, z


def probPCA(x_n, K=5, n_iter=500, noise=1.0):
    w = ed.models.Normal(tf.zeros((D, K)), tf.ones((D, K)))
    z = ed.models.Normal(tf.zeros((K, N)), tf.ones((K, N)))
    # qw = ed.models.Normal(tf.Variable(tf.random_normal(w.shape)),
    #                       tf.nn.softplus(tf.Variable(tf.random_normal(w.shape))))
    # qz = ed.models.Normal(tf.Variable(tf.random_normal(z.shape)),
    #                       tf.nn.softplus(tf.Variable(tf.random_normal(z.shape))))

    qw = ed.models.NormalWithSoftplusScale(tf.Variable(tf.random_normal(w.shape)),
                          tf.Variable(tf.random_normal(w.shape)))
    qz = ed.models.NormalWithSoftplusScale(tf.Variable(tf.random_normal(z.shape)),
                          tf.Variable(tf.random_normal(z.shape)))

    KL = {w: qw, z: qz}

    if hasattr(noise, '__iter__') and noise[0] == 'point':
        s = tf.Variable(noise[1], dtype=tf.float32)
        s_print = s
    elif noise == 'full':
        # g = ed.models.Gamma(1e-5, 1e-5)
        ig = ed.models.InverseGamma(1e-5, 1e-5)
        # s = g**-.5 #tf.pow(g, -0.5)
        qg = ed.models.TransformedDistribution(
            ed.models.NormalWithSoftplusScale(tf.Variable(0.), tf.Variable(1.)),
            bijector=tf.contrib.distributions.bijectors.Exp())
        # KL.update({g: qg})
        # qg = ed.models.InverseGammaWithSoftplusConcentrationRate(
        #     tf.Variable(tf.random_normal(shape=())), tf.Variable(tf.random_normal(shape=())))
        KL.update({ig: qg})
        s = tf.sqrt(ig)
        # s_print = qg.mean()
    else: # if noise is a simple number
        s = tf.constant(noise)
        s_print = s

    x = ed.models.Normal(tf.matmul(w, z), s * tf.ones((5, 5000)))

    inference = ed.KLqp(KL, data={x: x_n})
    # edward doesn't seem to have context manager here, but it also doesn't seem
    # necessary to have a session if one uses .eval()
    # sess = ed.get_session()
    out = inference.run(n_iter=n_iter)
    y = tf.matmul(qw.mean(), qz.mean()).eval()

    try:
        print('estimated/exact noise: ', s_print.eval())
    except:
        pass
    return y, qz.mean().eval()

def vbpca(x_n, n_iter):
    z = bp.nodes.GaussianARD(0, 1, plates=(1, N), shape=(K, ))
    alpha = bp.nodes.Gamma(1e-5, 1e-5, plates=(K, ))
    w = bp.nodes.GaussianARD(0, alpha, plates=(D, 1), shape=(K, ))
    tau = bp.nodes.Gamma(1e-5, 1e-5)
    x = bp.nodes.GaussianARD(bp.nodes.SumMultiply('d,d->', z, w), tau)
    x.observe(x_n)
    q = bp.inference.VB(x, z, w, alpha, tau)
    w.initialize_from_random()
    q.update(repeat=n_iter)

    W = w.get_moments()[0].squeeze()
    Z = z.get_moments()[0].squeeze()

    print('alphas: ', alpha.get_moments()[0])
    print('estimated noise: ', tau.get_moments()[0])
    return W.dot(Z.T), Z.T


def critique(x_n, x_c, z, x, pcs):
    err = {
        'x_n': np.abs(x_n - x).sum(),
        'x_c': np.abs(x_c - x).sum(),
        'pc': np.abs(z - pcs).sum(),
    }
    print('reconstruction errors: ', err)


if __name__=='__main__':
    w = np.random.normal(0, 1, (D, K))
    z = np.random.normal(0, 1, (K, N))
    x_c = w.dot(z)                                  # without noise ("clean")
    x_n = x_c + np.random.normal(0, 1, (D, N))   # with noise

    from functools import partial
    cr = partial(critique, x_n, x_c, z)
