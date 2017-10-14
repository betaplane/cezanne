#!/usr/bin/env python
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
# import tensorflow as tf
# import edward as ed
import bayespy as bp

# N number of examples ('time')
# D dimension of example ('space')
# K number of principal components

def whitened_test_data(N=5000, D=5, K=5, s=1, missing=0):
    w = np.random.normal(0, 1, (D, K))
    z = np.random.normal(0, 1, (K, N))
    x = w.dot(z)
    p = detPCA(x, K)
    mask = np.zeros(x.shape).flatten()
    mask[np.random.randint(0, len(mask), round(missing * len(mask)))] = 1
    x1 = np.ma.masked_array(x.flatten(), mask).reshape(x.shape)
    return x, x1 + np.random.normal(0, s, (D, N)), p.w, p.z


def tf_rotate(w, z):
    u, U = tf.self_adjoint_eig(tf.matmul(z, tf.matrix_transpose(z)))
    Dx = tf.diag(v ** .5)
    v, V = tf.self_adjoint_eig(
        tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.matmul(
            Dx, tf.matrix_transpose(U)), w), tf.matrix_transpose(w)), U), Dx))
    w_rot = tf.matmul(tf.matmul(tf.matmul(tf.matrix_transpose(w), U), Dx), V)
    z_rot = tf.matmul(tf.matmul(tf.matmul(tf.matrix_transpose(V), tf.diag(v ** -.5)),
                                tf.matrix_transpose(U)), z)
    return w_rot, z_rot


class PCA(object):
    def critique(self, x0, w0, z0, rotate=False):
        (w, z) = self.rotate() if rotate else (self.w, self.z)
        x = w.dot(z)

        # here we scale and sign the pc's and W matrix to facilitate compariso with the originals
        # Note: in deterministic PCA, e will be an array of ones, s.t. potentially the sorting (i) may be undetermined and mess things up.
        if not isinstance(self, detPCA):
            e = (w ** 2).sum(0)**.5
            i = np.argsort(e)[::-1]
            w = w[:, i]
            e = e[i].reshape((1, -1))
            s = np.sign(np.sign(w0[:, :w.shape[1]] * w).sum(0, keepdims=True))
            w = w / e * s
            z = z[i, :] * e.T * s.T
        self.w_rot = w

        err = {
            'noise': np.abs(x1 - x).sum(),
            'clean': np.abs(x0 - x).sum(),
            'W': np.abs(w0[:, :w.shape[1]] - w).sum(),
            'Z': np.abs(z0[:z.shape[0], :] - z).sum()
        }
        print('reconstruction errors: ', err)
        return self

        # Note: this simplified rotation applies only to probabilistic PCA, which already has some of the scalings built in.
    def rotate(self):
        e, v = np.linalg.eigh(self.w.T.dot(self.w))
        w_rot = self.w.dot(v)
        z_rot = v.T.dot(self.z)
        return w_rot, z_rot


class detPCA(PCA):
    def __init__(self, x1, K=5):
        self.e, v = np.linalg.eigh(np.cov(x1))
        self.w = v[:, np.argsort(self.e)[::-1][:K]]
        self.z = self.w.T.dot(x1)


class probPCA(PCA):
    def __init__(self, x1, K=5, n_iter=500, noise=1.0):
        self.x1 = x1
        D, N = x1.shape
        W = ed.models.Normal(tf.zeros((D, K)), tf.ones((D, K)))
        Z = ed.models.Normal(tf.zeros((K, N)), tf.ones((K, N)))

        QW = ed.models.NormalWithSoftplusScale(tf.Variable(tf.random_normal(W.shape)),
                              tf.Variable(tf.random_normal(W.shape)))
        QZ = ed.models.NormalWithSoftplusScale(tf.Variable(tf.random_normal(Z.shape)),
                              tf.Variable(tf.random_normal(Z.shape)))

        KL = {W: QW, Z: QZ}

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

        X = ed.models.Normal(tf.matmul(W, Z), s * tf.ones((D, N)))

        inference = ed.KLqp(KL, data={X: x1})
        # edward doesn't seem to have context manager here, but it also doesn't seem
        # necessary to have a session if one uses .eval()
        out = inference.run(n_iter=n_iter)
        # y = tf.matmul(qw.mean(), qz.mean()).eval()

        sess = ed.get_session()
        self.w, self.z = sess.run([QW.mean(), QZ.mean()])
        # rotate(QW.mean().eval(), QZ.mean().eval())
        # self.x = w.dot(z)

        try:
            print('estimated/exact noise: ', s_print.eval())
        except:
            pass

class vbPCA(PCA):
    def __init__(self, x1, K, n_iter):
        D, N = x1.shape
        z = bp.nodes.GaussianARD(0, 1, plates=(1, N), shape=(K, ))
        alpha = bp.nodes.Gamma(1e-5, 1e-5, plates=(K, ))
        w = bp.nodes.GaussianARD(0, alpha, plates=(D, 1), shape=(K, ))
        tau = bp.nodes.Gamma(1e-5, 1e-5)
        x = bp.nodes.GaussianARD(bp.nodes.SumMultiply('d,d->', z, w), tau)
        x.observe(x1)
        q = bp.inference.VB(x, z, w, alpha, tau)
        w.initialize_from_random()
        q.update(repeat=n_iter)

        self.w = w.get_moments()[0].squeeze()
        self.z = z.get_moments()[0].squeeze().T

        print('alphas: ', alpha.get_moments()[0])
        print('estimated noise: ', tau.get_moments()[0])



if __name__=='__main__':
    x0, x1, W, Z = whitened_test_data(5000, 5, 5, 1)
