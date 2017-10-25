#!/usr/bin/env python
"""
Notes
-----

* :class:`tf.contrib.distributions.GammaWithSoftplusConcentrationRate` produced negative 'concentration'. I therefore went with using :class:`tf.nn.softplus`, also in the case of :class:`ed.models.Normal` (instead of :class:`ed.models.NormalWithSoftplusScale`).
* The black-box :class:`ed.inference.KLqp` algorithm used by Edward (score function gradient) doesn't deal well with Gamma and Dirichlet:
    * https://github.com/blei-lab/edward/issues/389
    * https://gist.github.com/pwl/2f3c3e240b477eac9a37b06791b2a659
* For the tensorboard summaries to work in the presence of missing values, the input array needs to be of :class:`np.ma.MaskedArray` type **and** have NaNs at the missing locations - not clear why.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import edward as ed
import bayespy as bp
import bayespy.inference.vmp.transformations as bpt
from types import MethodType

# N number of examples ('time')
# D dimension of example ('space')
# K number of principal components

def station_data():
    t = pd.read_hdf('../../data/CEAZAMet/station_data.h5', 'ta_c').xs('prom', 1, 'aggr')
    sta = pd.read_hdf('../../data/CEAZAMet/stations.h5', 'stations')
    lat = sta.loc[t.columns.get_level_values(0)].lat.astype(float)
    return t[t.columns[(lat>-34) & (lat<-27)]].resample('D').mean()

def whitened_test_data(N=5000, D=5, K=5, s=1, missing=0):
    w = np.random.normal(0, 1, (D, K))
    z = np.random.normal(0, 1, (K, N))
    m = np.random.normal(0, 1, (D, 1))
    x = w.dot(z)
    p = detPCA(x, n_iter=1)
    x1 = x.flatten()
    x1[np.random.randint(0, len(x1), round(missing * len(x1)))] = np.nan # see notes in module docstring
    x1 = np.ma.masked_invalid(x1).reshape(x.shape)
    return x + m, x1 + m + np.random.normal(0, s, (D, N)), p.w, p.z, m


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
    def critique(self, x0=None, w0=None, z0=None, rotate=False):
        (w, z) = self.rotate() if rotate else (self.w, self.z)
        self.x = w.dot(z) + self.m

        # here we scale and sign the pc's and W matrix to facilitate comparison with the originals
        # Note: in deterministic PCA, e will be an array of ones, s.t. potentially the sorting (i) may be undetermined and mess things up.
        if not isinstance(self, detPCA):
            e = (w ** 2).sum(0)**.5
            i = np.argsort(e)[::-1]
            w = w[:, i]
            e = e[i].reshape((1, -1))
            w = w / e
            z = z[i, :] * e.T
            if w0 is not None:
                s = np.sign(np.sign(w0[:, :w.shape[1]] * w).sum(0, keepdims=True))
                w = w * s
                z = z * s.T
        self.w_rot = w

        print('\nnoisy input: ', ((self.x1 - self.x) ** 2).mean())
        print('data_loss: ', self.sess.run(self.data_loss))
        if x0 is not None: print('clean input: ', ((x0 - self.x) ** 2).mean())
        if w0 is not None: print('W: ', np.abs(w0[:, :w.shape[1]] - w).sum())
        if z0 is not None: print('Z: ', np.abs(z0[:z.shape[0], :] - z).sum())
        return self

    # Note: this simplified rotation applies only to probabilistic PCA, which already has some of the scalings built in.
    def rotate(self):
        e, v = np.linalg.eigh(self.w.T.dot(self.w))
        w_rot = self.w.dot(v)
        z_rot = v.T.dot(self.z)
        return w_rot, z_rot


class detPCA(PCA):
    def __init__(self, x1, K=None, n_iter=10):
        self.x1 = x1
        if K is None:
            K = x1.shape[0]
        mask = np.isnan(x1)
        self.m = x1.mean(1, keepdims=True)
        self.x = self.m
        for i in range(n_iter):
            x = np.where(mask, self.x, x1) - self.m
            self.e, v = np.linalg.eigh(np.cov(x))
            self.w = v[:, np.argsort(self.e)[::-1][:K]]
            self.z = self.w.T.dot(x)
            self.x = self.w.dot(self.z)
            self.m = self.x.mean(1, keepdims=True)
            loss = ((self.x - x) ** 2).sum()
            print(loss)

class PPCA(PCA):
    def __init__(self, x1, **kwargs):
        self.x1 = x1.copy()
        self.D, self.N = x1.shape
        self.dims = kwargs.get('dims', None)
        self.K = self.D if (self.dims is None or isinstance(self.dims, str)) else self.dims
        self.seed = kwargs.get('seed', None)
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        tf.set_random_seed(self.seed)
        self.loc = kwargs.get('loc', 'random_normal')

    def param_init(self, shape, name, values=None, kind='loc'):
        if self.loc == 'fixed':
            return tf.Variable(tf.zeros(shape), name='{}/{}'.format(name, kind))
        else:
            v = tf.Variable(getattr(tf, self.loc)(shape, seed=self.seed), name='{}/{}'.format(name, kind))
            return tf.nn.softplus(v) if kind == 'scale' else v


class gradPCA(PPCA):
    def __init__(self, x1, learning_rate, n_iter=100, **kwargs):
        super().__init__(x1, **kwargs)
        mask = 1 - np.isnan(x1).astype(int).filled(1)
        mask_sum = np.sum(mask, 1, keepdims=True)
        data_mean = x1.mean(1, keepdims=True)
        data = (x1 - data_mean).flatten()
        i, = np.where(~np.isnan(data))
        data = data[i]

        W = self.param_init((self.D, self.K), 'W', kind='hyper')
        Z = self.param_init((self.N, self.K), 'Z', kind='hyper')
        x = tf.matmul(W, Z, transpose_b=True)
        m = tf.reduce_sum(x * mask, 1, keep_dims=True) / mask_sum
        self.data_loss = tf.losses.mean_squared_error(tf.gather(tf.reshape(x - m, [-1]), i), data)
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.data_loss)

        prog = ed.Progbar(n_iter)
        tf.global_variables_initializer().run()
        for j in range(n_iter):
            _, out = self.sess.run([opt, self.data_loss])
            prog.update(j, {'loss': out})

        self.w, self.z, self.m = self.sess.run([W, tf.matrix_transpose(Z), m + data_mean])


class probPCA(PPCA):
    def lognormal(self, shape=(), name='LogNormal'):
        lognormal = ed.models.TransformedDistribution(
            ed.models.Normal(
                # since we use this for variance-like variables
                tf.nn.softplus(tf.Variable(tf.random_normal(shape, seed=self.seed)), name='{}/loc'.format(name)),
                tf.nn.softplus(tf.Variable(tf.random_normal(shape, seed=self.seed)), name='{}/scale'.format(name)),
                name = '{}_Normal'.format(name)
            ), bijector = tf.contrib.distributions.bijectors.Exp(),
            name = '{}_LogNormal'.format(name)
        )
        lognormal.mean = MethodType(self.lognormalmean, lognormal)
        lognormal.variance = MethodType(self.lognormalvariance, lognormal)
        return lognormal

    @staticmethod
    def lognormalmean(self):
        ds = self.distribution
        xmu, xvar = tf.exp(ds.mean()), tf.exp(ds.variance())
        return xmu * xvar ** .5

    @staticmethod
    def lognormalvariance(self):
        ds = self.distribution
        xmu, xvar = tf.exp(ds.mean()), tf.exp(ds.variance())
        return xmu ** 2 * xvar * (xvar - 1)

    def print(self, *vars):
        print('')
        for v in vars:
            if not isinstance(v, str):
                print(v)
            elif hasattr(self, v):
                try:
                    print(v, getattr(self, v).mean().eval())
                except AttributeError:
                    print(v, getattr(self, v).eval())

    def __init__(self, x1, n_iter=500,
                 full_prior=[], full_posterior=[], zero_locs=False,
                 mean='point', noise='point',
                 logdir='log', **kwargs):
        super().__init__(x1, **kwargs)

        full_prior = set(full_posterior).union(full_prior)
        KL = {}

        if self.dims == 'full':
            alpha = ed.models.Gamma(1e-5 * tf.ones((1, self.D)), 1e-5 * tf.ones((1, self.D)), name='model/alpha')
            qa = self.lognormal(alpha.shape, 'posterior/alpha')
            KL.update({alpha: qa})
            self.alpha = qa
            a = alpha ** -.5
        elif self.dims == 'point':
            raise Exception('not implemented')
        else:
            a = tf.ones((1, self.K), name='alpha') # in hopes that this is correctly broadcast

        with tf.name_scope('model'):
            if 'Z' in full_prior:
                self.Z_scale = tf.nn.softplus(tf.Variable(tf.random_normal((self.K, self.K), seed=self.seed)), name='Z/scale')
                Z = ed.models.MultivariateNormalTriL(tf.zeros((self.N, self.K)), self.Z_scale, name='Z')
            else:
                Z = ed.models.Normal(tf.zeros((self.N, self.K), name='Z/loc'), tf.ones((self.N, self.K), name='Z/scale'), name='Z')

            if 'W' in full_prior:
                self.W_scale = tf.nn.softplus(tf.Variable(tf.random_normal((self.K, self.K), seed=self.seed)), name='W/scale')
                W = ed.models.MultivariateNormalTriL(tf.zeros((self.D, self.K)), a * self.W_scale, name='W')
            else:
                W = ed.models.Normal(tf.zeros((self.D, self.K)), a, name='W')

        def post(shape, name, covariance='fact', zeros=False):
            return {
                'full': ed.models.MultivariateNormalTriL,
                'fact': ed.models.Normal
            }[covariance]({
                True: tf.zeros(shape),
                False: tf.Variable(tf.random_normal(shape, seed=self.seed), name='{}/loc'.format(name)),
            }[zeros], tf.nn.softplus(
                tf.Variable(tf.random_normal({'full': (self.K, self.K), 'fact': shape}[covariance], seed=self.seed)
                            , name='{}/scale'.format(name))), name=name)

        with tf.name_scope('posterior'):
            QZ = post(Z.shape, 'Z', 'full' if 'Z' in full_posterior else 'fact', zero_locs)
            QW = post(W.shape, 'W', 'full' if 'W' in full_posterior else 'fact', zero_locs)

        KL.update({Z: QZ, W: QW})

        if noise == 'point':
            tau = tf.nn.softplus(tf.Variable(tf.random_normal((), seed=self.seed), name='tau'))
            self.tau = tau
        elif noise == 'full':
            s = ed.models.Gamma(1e-5, 1e-5, name='model/tau')
            qs = self.lognormal(name='posterior/tau')
            tau = s ** -.5
            KL.update({s: qs})
            self.tau = qs
        else: # if noise is a simple number
            tau = tf.constant(noise)

        data_mean = x1.mean(1, keepdims=True).astype('float32')
        if mean == 'full':
            hyper_mean = tf.Variable(data_mean, name='hyper_mean')
            m = ed.models.Normal(hyper_mean, tf.ones((self.D, 1)))
            self.qm = ed.models.Normal(
                tf.Variable(data_mean),
                # tf.Variable(tf.random_normal((self.D, 1), seed=self.seed)),
                tf.nn.softplus(tf.Variable(tf.random_normal((self.D, 1), seed=self.seed))))
            KL.update({m: self.qm})
            mu  = hyper_mean #self.qm.mean()
        else:
            # m = tf.Variable(tf.random_normal((self.D, 1), seed=self.seed), name='mean')
            m = tf.Variable(data_mean, name='mean')
            mu = m

        data = x1.flatten()
        i, = np.where(~np.isnan(data))
        data = data[i]
        x = tf.gather(tf.reshape(tf.matmul(W, Z, transpose_b=True) + m, [-1]), i)
        X = ed.models.Normal(x, tau * tf.ones(x.shape))

        self.inference = ed.KLqp(KL, data={X: data})
        self.inference.initialize(n_iter=n_iter, n_samples=10, logdir=logdir)
        tf.global_variables_initializer().run()

        with tf.variable_scope('loss'):
            self.y = tf.matmul(QW.mean(), QZ.mean(), transpose_b=True) + m
            self.data_loss = tf.losses.mean_squared_error(data, tf.gather(tf.reshape(self.y, [-1]), i))
            # self.data_loss = tf.reduce_mean((data - tf.gather(tf.reshape(self.y, [-1]), i)) ** 2)
            tf.summary.scalar('data_loss', self.data_loss)
        self.merged = tf.summary.merge_all()

        # self.out = self.inference.run(n_iter=n_iter, n_samples=10, logdir=logdir)
        self.run(n_iter=n_iter)

        self.w, self.z, self.m = self.sess.run([QW.mean(), tf.matrix_transpose(QZ.mean()), mu])
        self.print('tau', 'alpha', self.m)

    def run(self, n_iter):
        for i in range(n_iter):
            out = self.inference.update()
            self.inference.print_progress(out)
            self.inference.train_writer.add_summary(self.sess.run(self.merged), i)


class vbPCA(PCA):
    def __init__(self, x1, K=None, n_iter=100, rotate=False):
        self.D, self.N = x1.shape
        self.x1 = x1

        z = bp.nodes.GaussianARD(0, 1, plates=(1, self.N), shape=(K, ))
        alpha = bp.nodes.Gamma(1e-5, 1e-5, plates=(K, ))
        w = bp.nodes.GaussianARD(0, alpha, plates=(self.D, 1), shape=(K, ))
        m = bp.nodes.GaussianARD(0, 1, shape=(self.D, 1))
        tau = bp.nodes.Gamma(1e-5, 1e-5)
        x = bp.nodes.GaussianARD(bp.nodes.Add(bp.nodes.Dot(z, w), m), tau)
        x.observe(x1, mask=~x1.mask)
        q = bp.inference.VB(x, z, w, alpha, tau, m)

        if rotate:
            rot_z = bpt.RotateGaussianARD(z)
            rot_w = bpt.RotateGaussianARD(w, alpha)
            R = bpt.RotationOptimizer(rot_z, rot_w, K)
            q.set_callback(R.rotate)

        w.initialize_from_random()
        q.update(repeat=n_iter)

        self.w = w.get_moments()[0].squeeze()
        self.z = z.get_moments()[0].squeeze().T
        self.mu = m.get_moments()[0].squeeze()
        self.tau = tau

        print('alphas: ', alpha.get_moments()[0])
        print('estimated noise: ', tau.get_moments()[0])



if __name__=='__main__':
    x0, x1, W, Z, m = whitened_test_data(5000, 5, 5, 1, missing=.3)
    # t = station_data()[['3','4','5','6','8','9']]
    # x = np.ma.masked_invalid(t).T
    pass
