#!/usr/bin/env python
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import tensorflow as tf
import edward as ed
import bayespy as bp
import bayespy.inference.vmp.transformations as bpt

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

        # here we scale and sign the pc's and W matrix to facilitate comparison with the originals
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
            'noise': np.abs(self.x1 - x).sum(),
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
        self.x1 = x1
        self.e, v = np.linalg.eigh(np.cov(x1))
        self.w = v[:, np.argsort(self.e)[::-1][:K]]
        self.z = self.w.T.dot(x1)


class probPCA(PCA):
    @staticmethod
    def lognormal():
        return ed.models.TransformedDistribution(
            ed.models.Normal(
                # since we use this for variance-like variables
                tf.nn.softplus(tf.Variable(tf.random_normal(()))),
                tf.nn.softplus(tf.Variable(tf.random_normal(())))
            ), bijector = tf.contrib.distributions.bijectors.Exp(),
            name = 'LogNormal'
        )

    @staticmethod
    def lognormalmean(ds):
        xmu, xvar = tf.exp(ds.mean()), tf.exp(ds.variance())
        return xmu * xvar ** .5

    @staticmethod
    def lognormalvariance(ds):
        xmu, xvar = tf.exp(ds.mean()), tf.exp(ds.variance())
        return xmu ** 2 * xvar * (xvar - 1)

    def __init__(self, x1, K=5, n_iter=500, noise='point', full_cov=False):
        self.x1 = x1
        D, N = x1.shape
        W = ed.models.Normal(tf.zeros((D, K)), tf.ones((D, K)))
        if full_cov:
            self.cov = tf.nn.softplus(tf.Variable(tf.random_normal((K, K))))
            Z = ed.models.MultivariateNormalTriL(tf.zeros((N, K)), self.cov)
        else:
            Z = ed.models.Normal(tf.zeros((N, K)), tf.ones((N, K)))

        # somewhat skeptical about the NormalWithSoftplusScale method (the GammaWithSoftplusConcentrationRate resulted in negative concentration for some reason)
        QW = ed.models.Normal(tf.Variable(tf.random_normal(W.shape)),
                             tf.nn.softplus(tf.Variable(tf.random_normal(W.shape))))
        QZ = ed.models.Normal(tf.Variable(tf.random_normal(Z.shape)),
                              tf.nn.softplus(tf.Variable(tf.random_normal(Z.shape))))

        KL = {W: QW, Z: QZ}

        if noise == 'point':
                tau = tf.nn.softplu(tf.Variable(tf.random_normal(()), dtype=tf.float32))
                tau_print = tau
                self.tau = tau
        elif noise == 'full':
            # the black-box KLqp algorithm used by Edward (score function gradient) doesn't deal well with Gamma and Dirichlet:
            # https://github.com/blei-lab/edward/issues/389
            # https://gist.github.com/pwl/2f3c3e240b477eac9a37b06791b2a659
            s = ed.models.Gamma(1e-5, 1e-5)
            qs = self.lognormal()
            tau = s ** -.5
            tau_print = self.lognormalmean(qs.distribution)
            KL.update({s: qs})
            self.tau = qs
        else: # if noise is a simple number
            tau = tf.constant(noise)
            tau_print = tau

        X = ed.models.Normal(tf.matmul(W, Z, transpose_b=True), tau * tf.ones((D, N)))

        self.inference = ed.KLqp(KL, data={X: x1})
        self.out = self.inference.run(n_iter=n_iter, n_samples=10)

        sess = ed.get_session()
        self.w, self.z = sess.run([QW.mean(), tf.matrix_transpose(QZ.mean())])

        try:
            print('estimated/exact noise: ', tau_print.eval())
        except:
            pass

class vbPCA(PCA):
    def __init__(self, x1, K, n_iter, rotate=False):
        D, N = x1.shape
        self.x1 = x1
        z = bp.nodes.GaussianARD(0, 1, plates=(1, N), shape=(K, ))
        alpha = bp.nodes.Gamma(1e-5, 1e-5, plates=(K, ))
        w = bp.nodes.GaussianARD(0, alpha, plates=(D, 1), shape=(K, ))
        tau = bp.nodes.Gamma(1e-5, 1e-5)
        x = bp.nodes.GaussianARD(bp.nodes.SumMultiply('d,d->', z, w), tau)
        x.observe(x1, mask=~x1.mask)
        q = bp.inference.VB(x, z, w, alpha, tau)

        if rotate:
            rot_z = bpt.RotateGaussianARD(z)
            rot_w = bpt.RotateGaussianARD(w, alpha)
            R = bpt.RotationOptimizer(rot_z, rot_w, K)
            q.set_callback(R.rotate)

        w.initialize_from_random()
        q.update(repeat=n_iter)

        self.w = w.get_moments()[0].squeeze()
        self.z = z.get_moments()[0].squeeze().T
        self.tau = tau

        print('alphas: ', alpha.get_moments()[0])
        print('estimated noise: ', tau.get_moments()[0])



if __name__=='__main__':
    # x0, x1, W, Z = whitened_test_data(5000, 5, 5, 1)
    pass
