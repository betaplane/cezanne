"""
Notes
-----
* The instance variables exposed by the various classes have the following meaning:
    * **W** (D, K) - the weights / loadings matrix_transpose
    * **Z** (N, K) - the principal components
    * **mu** (D, 1) - the data means (per dimension)
    * **x** (D, N) - the reconstructed data
* :class:`tf.contrib.distributions.GammaWithSoftplusConcentrationRate` produced negative 'concentration'. I therefore went with using :class:`tf.nn.softplus`, also in the case of :class:`ed.models.Normal` (instead of :class:`ed.models.NormalWithSoftplusScale`).
* The black-box :class:`ed.inference.KLqp` algorithm used by Edward (score function gradient) doesn't deal well with Gamma and Dirichlet:
    * https://github.com/blei-lab/edward/issues/389
    * https://gist.github.com/pwl/2f3c3e240b477eac9a37b06791b2a659
* For the tensorboard summaries to work in the presence of missing values, the input array needs to be of :class:`np.ma.MaskedArray` type **and** have NaNs at the missing locations - not clear why.

ToDo
----
* separate data observation from graph initialization in probPCA (use tf.placeholder)
* write out the loss function value internally used by the algorithms in the losses dataframe

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import edward as ed
import bayespy as bp
import bayespy.inference.vmp.transformations as bpt
from types import MethodType
from datetime import datetime
from warnings import warn, catch_warnings, simplefilter
import os

# N number of examples ('time')
# D dimension of example ('space')
# K number of principal components



class PCA(object):
    """Base class for all PCA subtypes. Any **kwargs** passed to the constructor will be added to the :attr:`losses` accumulator. The idea is to automatically save any configuration values for later analysis of the accumulated results.
    """

    losses = pd.DataFrame()
    """Class-level accumulator :class:`~pandas.DataFrame` for all data that's supposed to be saved for a given invocation."""

    def __getattr__(self, name):
        try:
            v = self.variables[name]
        except KeyError:
            warn('Attempted to access non-existing variable.')
            return np.nan # so that in summary DataFrame we have NaN instead of None
        else:
            return self.sess.run(v) if isinstance(self, PPCA) else v

    def __init__(self, **kwargs):
        self.variables = {}
        self.instance = kwargs
        self.instance.update({
            'id': datetime.utcnow().strftime('pca%Y%m%d%H%M%S%f'),
            'class': self.__class__.__name__
        })

    def append(self, **kwargs):
        kwargs.update({'logsubdir': self.logsubdir, 'n_iter': self.n_iter})
        kwargs.update(self.instance)
        PCA.losses = PCA.losses.append(kwargs, ignore_index=True)

    def critique(self, data=None, x0=None, w0=None, z0=None, rotate=False):
        (w, z) = self.rotate() if rotate else (self.W, self.Z)

        # here we scale and sign the pc's and W matrix to facilitate comparison with the originals
        # Note: in deterministic PCA, e will be an array of ones, s.t. potentially the sorting (i) may be undetermined and mess things up.
        if not isinstance(self, detPCA):
            e = (w ** 2).sum(0)**.5
            i = np.argsort(e)[::-1]
            w = w[:, i]
            e = e[i].reshape((1, -1))
            w = w / e
            z = z[:, i] * e
            if hasattr(data, 'W'):
                s = np.sign(np.sign(data.W[:, :w.shape[1]] * w).sum(0, keepdims=True))
                w = w * s
                z = z * s

        print('')
        # because this is precicely when the warning raised in __getattr__() should be ignored
        with catch_warnings():
            simplefilter('ignore')
            for v in ['data_loss', 'tau', 'alpha']:
                print('{}: {}'.format(v, getattr(self, v)))

        if data is not None:
            update = {a: self.RMS(data, a, W=w, Z=z) for a in ['x', 'W', 'Z', 'mu', 'tau']}
            update.update({'missing': data.missing_fraction, 'data': data.id})

        update.update({'data_loss': self.data_loss, 'rotated': rotate})
        self.append(**update)

        return self

    def RMS(self, data, attr, **kwargs):
        # transforming to df and resetting index serves as an alignment tool for differing D and K
        def df(x):
            return pd.DataFrame(x).reset_index(drop=True) if hasattr(x, '__iter__') else x
        a = df(getattr(data, attr))
        try:
            d = (a - df(kwargs.get(attr, getattr(self, attr)))) ** 2
            return d.as_matrix().mean() ** .5 if hasattr(d, '__iter__') else d ** .5
        except:
            pass

class detPCA(PCA):
    def run(self, x1, K=None, n_iter=1):
        if K is None:
            K = x1.shape[0]
        data_mean = x1.mean(1, keepdims=True)
        x1 = (x1 - data_mean)
        x = x1.filled(0)
        for i in range(n_iter):
            self.e, v = np.linalg.eigh(np.cov(x))
            self.W = v[:, np.argsort(self.e)[::-1][:K]]
            self.Z = x.T.dot(self.W)
            self.x = self.W.dot(self.Z.T)
            diff = x1 - x
            m = diff.mean(1, keepdims=True)
            x = np.where(x1.mask, self.x + m, x1)
        self.mu = data_mean + m
        self.x = self.x + self.mu
        return self

class PPCA(PCA):
    """Parent class for probabilistic PCA subclasses that need TensorFlow_ infrastructure."""

    def __init__(self, shape, **kwargs):
        self.D, self.N = shape
        self.__dict__.update({key: kwargs.get(key) for key in ['dims', 'seed', 'logdir', 'n_iter']})
        self.dtype = kwargs.get('dtype', tf.float32)
        self.K = self.D if (self.dims is None or isinstance(self.dims, str)) else self.dims
        super().__init__(**kwargs) # additional kwargs are used to annotate the 'losses' DataFrame

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        tf.set_random_seed(self.seed)

    def get_config(self, *args):
        c = getattr(self, args[0], None)
        c = c.get(tf.get_variable_scope().name, c) if isinstance(c, dict) else c
        for a in args[1:]:
            if not isinstance(c, dict):
                break
            c = c.get(a, c) # this allows skipping of non-existant keys
        return c

    def param_init(self, name):
        n, kind = name.split('/')
        scope = tf.get_variable_scope() # e.g. posterior
        train = self.get_config('trainable', kind, n)
        init = self.get_config('initializer', kind, train)
        try:
            init = init(seed = self.seed)
        except TypeError:
            init = init()

        if kind=='scale' and self.get_config('covariance', kind, n)=='full':
            shape = (self.K, self.K)
        else:
            shape = {'Z': (self.N, self.K), 'W': (self.D, self.K)}[n]

        v = tf.get_variable(name, shape, self.dtype, init, trainable=train)
        return tf.nn.softplus(v) if kind == 'scale' else v

    def rotate(self):
        e, v = np.linalg.eigh(self.W.T.dot(self.W))
        w_rot = self.W.dot(v) # W ~ (D, K)
        z_rot = self.Z.dot(v) # Z ~ (N, K)
        return w_rot, z_rot

    @property
    def logsubdir(self):
        try:
            return os.path.split(self.inference.train_writer.get_logdir())[1]
        except AttributeError:
            return None

    @logsubdir.setter
    def logsubdir(self, value):
        if self.logdir is not None:
            self.inference.train_writer = tf.summary.FileWriter(os.path.join(self.logdir, value))


class gradPCA(PPCA):
    trainable = True
    initializer = tf.zeros_initializer# tf.random_normal_initializer

    def __init__(self, x1, learning_rate, n_iter=100, **kwargs):
        super().__init__(x1.shape, **kwargs)
        mask = 1 - np.isnan(x1).astype(int).filled(1)
        mask_sum = np.sum(mask, 1, keepdims=True)
        data_mean = x1.mean(1, keepdims=True)
        data = (x1 - data_mean).flatten()
        i, = np.where(~np.isnan(data))
        data = data[i]

        p = detPCA(x1, n_iter=1)

        W = self.param_init('W/hyper') + p.w
        Z = self.param_init('Z/hyper') + p.z.T
        x = tf.matmul(W, Z, transpose_b=True)
        m = tf.reduce_sum(x * mask, 1, keep_dims=True) / mask_sum
        self.data_loss = tf.losses.mean_squared_error(tf.gather(tf.reshape(x - m, [-1]), i), data)
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.data_loss)

        prog = ed.Progbar(n_iter)
        tf.global_variables_initializer().run()

        if self.logdir is not None:
            tf.summary.scalar('data_loss', self.data_loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                os.path.join(self.logdir, datetime.utcnow().strftime('%Y%m%d_%H%M%S')))

            for j in range(n_iter):
                _, out, s = self.sess.run([opt, self.data_loss, merged])
                prog.update(j, {'loss': out})
                writer.add_summary(s, j)
        else:
            for j in range(n_iter):
                _, out = self.sess.run([opt, self.data_loss])
                prog.update(j, {'loss': out})

        mu = m + data_mean
        self.W, self.Z, self.mu, self.x = self.sess.run([W, Z, mu, x + mu])


class probPCA(PPCA):
    """Edward_-based fully configurable bayesian / mixed probabilistic principal component analyzer."""

    def lognormal(self, name, shape=()):
        lognormal = ed.models.TransformedDistribution(
            ed.models.Normal(
                # since we use this for variance-like variables
                tf.nn.softplus(
                    tf.get_variable('{}/loc'.format(name), shape=shape,
                                    initializer=self.get_config('initializer', name))),
                tf.nn.softplus(
                    tf.get_variable('{}/scale'.format(name), shape=shape,
                                    initializer=self.get_config('initializer', name))),
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


    trainable = {
        'posterior': True,
        'prior': {
            'loc': False,
            'scale': False #{'W': False, 'Z': False}
        }
    }
    """The trainable components of the PPCA system."""

    initializer = {
        'posterior': tf.random_normal_initializer,
        'prior': {
            'loc': tf.zeros_initializer,
            'scale': {True: tf.random_normal_initializer, False: tf.ones_initializer}
        },
        'tau': tf.random_normal_initializer
    }
    """The initializers used for the trainable components."""

    covariance = {
        'posterior': 'fact',
        'prior': {
            'W': 'fact',
            'Z': 'fact'
        }
    }
    """The type of covariance (full or factorized) to be used for the respective components."""

    def __init__(self, x1, hyper, mean='point', noise='point', **kwargs):
        shape = x1.shape
        super().__init__(shape, **kwargs)
        KL = {}

        if self.dims == 'full':
            alpha = ed.models.Gamma(1e-5 * tf.ones((1, self.D)), 1e-5 * tf.ones((1, self.D)), name='model/alpha')
            qa = self.lognormal('posterior/alpha', alpha.shape)
            KL.update({alpha: qa})
            self.alpha = qa
            a = alpha ** -.5
        elif self.dims == 'point':
            raise Exception('not implemented')
        else:
            a = tf.ones((1, self.K), name='alpha') # in hopes that this is correctly broadcast

        def normal(name):
            return {
                'full': ed.models.MultivariateNormalTriL,
                'fact': ed.models.Normal
            }[self.get_config('covariance', name)](
                *[self.param_init('{}/{}'.format(name, k)) for k in ['loc', 'scale']], name=name)

        with tf.variable_scope('prior'):
            Z = normal('Z')
            if self.get_config('covariance', 'W') == 'full':
                W = ed.models.MultivariateNormalTriL(tf.zeros((self.D, self.K)), a * self.param_init('W/scale'), name='W')
            else:
                W = ed.models.Normal(tf.zeros((self.D, self.K)), a, name='W')

        with tf.variable_scope('posterior'):
            QZ, QW = map(normal, ['Z', 'W'])

        KL.update({Z: QZ, W: QW})

        with tf.variable_scope(''):
            data_mean = tf.cast(x1.mean(1, keepdims=True), self.dtype)
            if mean == 'point':
                m = tf.get_variable('posterior/mu/loc', initializer=data_mean)
                self.variables.update({'mu': m})
            elif mean == 'full':
                hyper_mean = tf.get_variable('posterior/mu/hyper', initializer=data_mean) if hyper else data_mean
                m = ed.models.Normal(hyper_mean, tf.ones((self.D, 1)))
                qm = ed.models.Normal(
                    tf.Variable(data_mean),
                    tf.nn.softplus(tf.Variable(tf.random_normal((self.D, 1), seed=self.seed))))
                KL.update({m: qm})
                self.variables.update({'mu': qm.mean()})

            if noise == 'point':
                init = self.get_config('initializer', 'tau')
                tau = tf.nn.softplus(
                    tf.get_variable('posterior/tau', shape=(), initializer=init(self.seed)))
                self.variables.update({'tau': tau})
            elif noise == 'full':
                s = ed.models.Gamma(1e-5, 1e-5, name='prior/tau')
                qs = self.lognormal(name='posterior/tau')
                tau = s ** -.5
                KL.update({s: qs})
                self.variables.update({'tau': qs.mean()})

        data = x1.flatten()
        i, = np.where(~np.isnan(data))
        data = data[i]
        x = tf.gather(tf.reshape(tf.matmul(W, Z, transpose_b=True) + m, [-1]), i)

        # self.data = tf.placeholder(self.dtype, shape, name='x1')
        # self.mask = tf.placeholder(tf.bool, shape, name='mask')
        # data = tf.boolean_mask(self.data, self.mask, name='masked_x1')
        # x = tf.boolean_mask(tf.matmul(W, Z, transpose_b=True) + m, self.mask)

        X = ed.models.Normal(x, tau * tf.ones(x.shape))

        self.inference = ed.KLqp(KL, data={X: data})

        # this comes from edward source (class VariationalInference)
        # this way, edward automatically writes out my own summaries
        summary_key = 'summaries_{}'.format(id(self.inference))

        # data_loss is not instrumental in the procedure, I compute it solely to write it out to tensorboard
        with tf.variable_scope('loss'):
            xm = tf.add(tf.matmul(QW.mean(), QZ.mean(), transpose_b=True), m, name='x')
            data_loss = tf.losses.mean_squared_error(data, tf.gather(tf.reshape(xm, [-1]), i))
            tf.summary.scalar('data_loss', data_loss, collections=[summary_key])

        self.inference.initialize(n_samples=10, logdir=self.logdir)
        self.variables.update({'x': xm, 'W': QW.mean(), 'Z': QZ.mean(), 'data_loss': data_loss})


    def run(self, n_iter):
        self.n_iter = n_iter
        # if this is a repeated run, replace edward's FileWriter to write to a new directory
        try:
            t = self.inference.t.eval()
        except tf.errors.FailedPreconditionError:
            pass
        else:
            self.logsubdir = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        tf.global_variables_initializer().run()

        # hack to use the progbar that edward allocates anyway, without giving n_iter to inference.initialize()
        self.inference.progbar.target = n_iter

        for i in range(n_iter):
            out = self.inference.update()
            self.inference.print_progress(out)

        self.inference.finalize() # this just closes the FileWriter
        return self


class vbPCA(PCA):
    def __init__(self, x1, K=None, n_iter=100, rotate=False):
        super().__init__(rotated=rotate)
        self.D, self.N = x1.shape
        K = self.D if K is None else K
        self.x1 = x1

        z = bp.nodes.GaussianARD(0, 1, plates=(1, self.N), shape=(K, ))
        alpha = bp.nodes.Gamma(1e-5, 1e-5, plates=(K, ))
        w = bp.nodes.GaussianARD(0, alpha, plates=(self.D, 1), shape=(K, ))
        m = bp.nodes.GaussianARD(0, 1, shape=(self.D, 1))
        tau = bp.nodes.Gamma(1e-5, 1e-5)
        x = bp.nodes.GaussianARD(bp.nodes.Add(bp.nodes.Dot(z, w), m), tau)
        x.observe(x1, mask=~x1.mask)
        self.inference = bp.inference.VB(x, z, w, alpha, tau, m)

        if rotate:
            rot_z = bpt.RotateGaussianARD(z)
            rot_w = bpt.RotateGaussianARD(w, alpha)
            R = bpt.RotationOptimizer(rot_z, rot_w, K)
            self.inference.set_callback(R.rotate)

        w.initialize_from_random()
        self.inference.update(repeat=n_iter)

        self.W = w.get_moments()[0].squeeze()
        self.Z = z.get_moments()[0].squeeze()
        self.mu = m.get_moments()[0]
        self.tau = tau.get_moments()[0].item()
        self.x = self.W.dot(self.Z.T) + self.mu
        self.alpha = alpha.get_moments()[0]
        self.n_iter = self.inference.iter
        self.loss = self.inference.loglikelihood_lowerbound()



if __name__=='__main__':
    # x0, x1, W, Z, m = whitened_test_data(5000, 5, 5, 1, missing=.3)
    # t = station_data()[['3','4','5','6','8','9']]
    # x = np.ma.masked_invalid(t).T
    pass
