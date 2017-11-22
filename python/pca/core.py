"""
Conventions
===========
* The instance variables exposed by the various classes have the following meaning:
    * **W** (D, K) - the weights / loadings matrix_transpose
    * **Z** (N, K) - the principal components
    * **mu** (D, 1) - the data means (per dimension)
    * **x** (D, N) - the reconstructed data

Notes for myself
================
* Numpy eigenvalues are indeed **not** sorted.

TensorFlow_
-----------
* :class:`tf.contrib.distributions.GammaWithSoftplusConcentrationRate` produced negative 'concentration'. I therefore went with using :class:`tf.nn.softplus`, also in the case of :class:`ed.models.Normal` (instead of :class:`ed.models.NormalWithSoftplusScale`).
* For the tensorboard summaries to work in the presence of missing values, the input array needs to be of :class:`np.ma.MaskedArray` type **and** have NaNs at the missing locations - not clear why.

Edward_
-------
* Edward constructs different type of approximations to the loss function to be optimized for variational Bayes. The summaries written to tensorboard (in the 'loss' scope) reveal if the variational approximation can be computed analytically or not:
    * **analytic KL** (both prior and variational approx. are normal for all latent variables):
        * p_log_lik
        * kl_penalty
    * **intractable KL**:
        * p_log_prob
        * q_log_prob
    The reparameterization gradient method is used if all :attr:`edward.inference.latent_vars` have the :attr:`tensorflow.contrib.distributions.FULLY_REPARAMETERIZED` reparameterization type, otherwise the score function gradient method is used.
* The black-box :class:`ed.inference.KLqp` algorithm used by Edward (score function gradient) doesn't deal well with Gamma and Dirichlet:
    * https://github.com/blei-lab/edward/issues/389
    * https://gist.github.com/pwl/2f3c3e240b477eac9a37b06791b2a659

.. todo::

    * what happens if the main data model (passed to KLqp) has full covariance?

PCA
---

"""

import pandas as pd
import numpy as np
import edward as ed
import tensorflow as tf
import bayespy as bp
import bayespy.inference.vmp.transformations as bpt
from types import MethodType
from datetime import datetime
from warnings import warn
import os, time

# N number of examples ('time')
# D dimension of example ('space')
# K number of principal components



class PCA(object):
    """Base class for all PCA subtypes. The :meth:`critique` method computes various loss measures and appends them as a :class:`~pandas.DataFrame` to an HDF5 file. **Any \*\*kwargs passed to the constructor are appended to the DataFrame as additional columns so as to allow identification of individual experiments.** 

    """

    def __init__(self, **kwargs):
        self.variables = {}
        self.instance = kwargs
        self.instance.update({'class': self.__class__.__name__})
        self.id = datetime.utcnow().strftime('pca%Y%m%d%H%M%S%f')
        """A unique ID to identify instances of this class, e.g. in results tables. Constructe from :meth:`~datetime.datetime.utcnow`."""

    def critique(self, data=None, rotate=True, file_name=None, table_name=None, row=None):
        """Compute various loss measures of the PCA reconstruction w.r.t. the original data if `data` is provided, otherwise juse print the training error (:attr:`train_loss`). Also optionally rotates the principal components `Z` and loadings `W` prior to comparison.

        :param data: An object holding various components of the original data for comparison.
        :type data: :class:`.Data`
        :param rotate: Whether or not to rotate the principal components (:meth:`PPCA.rotate` needs to be defined on subclass).
        :type rotate: :obj:`bool`
        :param file_name: File name to append results to.
        :param table_name: Name of able inside file ``file_name`` to which to append results. If either ``file_name`` or ``table_name`` or **not** given, nothing will be written out.
        :param row: Index to give the row to be appended. If none is giben, defaults to :attr:`id`.
        :returns: The instance of the :class:`PCA` subclass, for method chaining.
        :rtype: :class:`PCA` subclass.

        """
        (w, z) = self.rotate() if rotate else (self.W, self.Z)

        # here we scale and sign the pc's and W matrix to facilitate comparison with the originals
        # Note: in deterministic PCA, e will be an array of ones, s.t. potentially the sorting (i) may be undetermined and mess things up.
        try:
            w, z = self.scale(w, z)
        except Exception as e:
            raise e # while developing
        try:
            s = np.sign(np.sign(data.W[:, :w.shape[1]] * w).sum(0, keepdims=True))
            w = w * s
            z = z * s
        except Exception as e:
            raise e # while developing

        if data is not None:
            results = {a: self.RMS(data, a, W=w, Z=z) for a in ['x', 'W', 'Z', 'mu', 'tau']}
            results.update({'missing': data.missing_fraction, 'data_id': data.id})

        print('')
        for v in ['train_loss', 'tau', 'alpha']:
            print('{}: {}'.format(v, getattr(self, v, np.nan)))

        results.update({
            'train_loss': self.train_loss,
            'rotated': rotate,
            'loss': self.loss,
            'logs': self.logsubdir,
            'n_iter': self.n_iter,
            'convergence_test': self.convergence_test
        })

        results.update(self.instance)

        self.results = pd.DataFrame(results, index=[self.id] if row is None else [row])
        if (file_name is not None) and (table_name is not None):
            self.results.to_hdf(file_name, table_name, format='t', append=True)
        else:
            warn('No results file and/or table name specified - results not written to file.')
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
    """Parent class for probabilistic PCA subclasses that need TensorFlow_ infrastructure.

    :Keyword Arguments:
        All are optional.

        * **logdir** - If given, TensorBoard summaries are saved in the given directory, with sub-directories made up from timestamps (as per Edward_ defaults).
        * **seed** - A random seed.

        The following are at present only relevant for :class:`probPCA`.

        * **dims**
                * ``full`` - apply automatic relevance determination to the columns of the loadings matrix :attr:`W`
                * :obj:`int` - use this many dimensions in the principal component space

    .. attribute:: graph

        The TensorFlow_ `graph` constructed for any particular subclass. I am trying to separate graph construction and execution as far as possible, so that e.g. one can construct a `tf.Session` or `tf.InteractiveSession` outside the module with this graph.
    """

    def __init__(self, shape, **kwargs):
        self.D, self.N = shape
        self.__dict__.update({key: kwargs.get(key) for key in ['dims', 'seed', 'logdir']})
        self.dtype = kwargs.get('dtype', tf.float32)
        self.K = self.D if (self.dims is None or isinstance(self.dims, str)) else self.dims

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

        super().__init__(**kwargs) # additional kwargs are used to annotate the 'losses' DataFrame

    # NOTE: all the helper methods of this class probably need to be called within a specific graph context
    def _init(self, init, shape=None):
        """Exists because if ``initializer`` is a :class:`~tensorflow.Tensor` or convertible to one, the ``shape`` kyeword is not needed. If ``init`` is a :obj:`str`, retrieves the attribute of this name from the parent class and uses it as initializer.

        """
        if isinstance(init, str):
            return {'initializer': getattr(self, init)}
        return {'initializer': init(), 'shape': shape}

    def param_init(self, name, scope='', kind=''):
        if scope == '':
            scope = tf.get_variable_scope().name
        full_name = '/'.join(n for n in [scope, name, kind] if n != '')

        train, init = self.config.loc[(scope, name, kind), :]

        if kind == 'scale' and (self.model[name] == scope or self.model[name] == 'all'):
            shape = (self.K, self.K)
        else:
            shape = {'Z': (self.N, self.K), 'W': (self.D, self.K), 'mu': (self.D, 1), 'tau':()}[name]

        v = tf.get_variable(full_name, dtype=self.dtype, trainable=train, **self._init(init, shape))
        return tf.nn.softplus(v) if kind == 'scale' else v

    def rotate(self):
        """Rotate principal components :attr:`Z` and loadings matrix :attr:`W` to form an orthogonal set. (No normalization is applied at this point)."""
        e, v = np.linalg.eigh(self.W.T.dot(self.W))
        w_rot = self.W.dot(v) # W ~ (D, K)
        z_rot = self.Z.dot(v) # Z ~ (N, K)
        return w_rot, z_rot

    @staticmethod
    def scale(W, Z):
        e = (W ** 2).sum(0)**.5
        i = np.argsort(e)[::-1]
        e = e[i].reshape((1, -1))
        w_scaled = W[:, i] / e
        z_scaled = Z[:, i] * e
        return w_scaled, z_scaled

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

    @property
    def is_reparameterizable(self):
        """Taken from edward.inferences.klqp.KLqp.build_loss_and_gradients()."""
        return all([rv.reparameterization_type == tf.contrib.distributions.FULLY_REPARAMETERIZED
                    for rv in self.inference.latent_vars.values()])

    @property
    def is_analytic(self):
        """Taken from edward.inferences.klqp.KLqp.build_loss_and_gradients()."""
        return all([isinstance(z, ed.models.Normal) and isinstance(qz, ed.models.Normal)
                    for z, qz in self.inference.latent_vars.items()])

# NOTE: this class is completely out of date with the rest of the module and not expected to work
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

        W = self.param_init('W', kind='hyper') + p.w
        Z = self.param_init('Z', kind='hyper') + p.z.T
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
    """Edward_-based fully configurable bayesian / mixed probabilistic principal component analyzer.

    :param shape: Shape of the data to expect in the :meth:`run` method. Needs to be (D, N) (see `Conventions`_).
    :param config: A modified configuration DataFrame; the default can be obtained by a call to :meth:`configure`.

    :Keyword Arguments:
        Optional keywords specify the type of prior and posterior approximation to be used and should be used in conjunction with appropriate settings of the `config` DataFrame. The options are at this point either ``none`` (a :obj:`str`, not :obj:`None`) -- the default -- or ``full``. For the loadings matrix and the principal components, this refers, respectively, to a completely factorized Gaussian model or one with full covariance matrix. If the posterior approximation is ``full``, the prior is automatically ``full`` too. For the 'hyperparamters' :math:`\\tau` and :math:`\mu`, ``none`` means that the parameters are point-estimated, whereas ``full`` means that they are given a full Bayesian treatment.

        * **W** - The loadings matrix.
        * **Z** - The principal components.
        * **tau** - The noise level of the data, a scalar.
        * **mu** - The means of the data vectors, with shape (D, ).

        See also the keyword arguments to the parent classes :class:`PPCA` and :class:`PCA`.
    """

    def lognormal(self, name, scope='posterior', shape=()):
        lognormal = ed.models.TransformedDistribution(
            ed.models.Normal(
                # since we use this for variance-like variables
                tf.nn.softplus(
                    tf.get_variable('{}/loc'.format(name), **self._init(
                                        self.config.loc[(scope, name, 'loc'), 'initializer'], shape))),
                tf.nn.softplus(
                    tf.get_variable('{}/scale'.format(name), **self._init(
                                        self.config.loc[(scope, name, 'scale'), 'initializer'], shape))),
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

    @staticmethod
    def configure(display=False):
        """Return the default configuration :class:`~pandas.DataFrame`. If ``display=True``, return a version with python objects replaced by strings, e.g. to display it in a Jupyter notebook."""

        idx = pd.IndexSlice
        config = pd.DataFrame(
            index = pd.MultiIndex.from_product([['prior', 'posterior'], ['W', 'Z', 'tau', 'mu'], ['loc', 'scale']]),
            columns = ['trainable', 'initializer']
        ).sort_index()
        config.loc[idx['prior', :, 'loc'], :]      = [False, tf.zeros_initializer]
        config.loc[idx['prior', :, 'scale'], :]    = [False, tf.ones_initializer]
        config.loc[idx['prior', 'mu', 'loc'], :]   = [True, 'data_mean']
        config.loc[idx['prior', 'mu', 'scale'], :] = [True, tf.random_normal_initializer]
        config.loc['posterior', :]                 = [True, tf.random_normal_initializer]
        if display:
            return config.replace({
                tf.random_normal_initializer: 'random',
                tf.ones_initializer: 'ones',
                tf.zeros_initializer: 'zeros'
            })
        return config

    def __init__(self, shape, config=None, test_data=False, **kwargs):
        model = {'W': 'none', 'Z': 'none', 'mu': 'full', 'tau': 'full'}
        self.model = {k: kwargs.pop(k, model[k]) for k in ['W', 'Z', 'mu', 'tau']}
        super().__init__(shape, **kwargs)
        self.config = self.configure() if config is None else config
        KL = {}

        with self.graph.as_default():
            if self.dims == 'full':
                alpha = ed.models.Gamma(1e-5 * tf.ones((1, self.D)), 1e-5 * tf.ones((1, self.D)), name='prior/alpha')
                qa = self.lognormal('posterior/alpha', alpha.shape)
                KL.update({alpha: qa})
                self.alpha = qa
                a = alpha ** -.5
            elif self.dims == 'point':
                raise Exception('not implemented')
            else:
                a = tf.ones((1, self.K), name='alpha') # in hopes that this is correctly broadcast

            def normal(name):
                scope = tf.get_variable_scope().name
                return {
                    True: ed.models.MultivariateNormalTriL,
                    False: ed.models.Normal
                }[self.model[name] == scope or self.model[name] == 'all'](
                    *[self.param_init(name, kind=k) for k in ['loc', 'scale']], name='{}/{}'.format(scope, name))

            with tf.variable_scope('prior'):
                Z = normal('Z')
                if self.model['W'] == 'prior' or self.model['W'] == 'all':
                    W = ed.models.MultivariateNormalTriL(tf.zeros((self.D, self.K)),
                                                         a * self.param_init('W', kind='scale'), name='W')
                else:
                    W = ed.models.Normal(tf.zeros((self.D, self.K)), a, name='W')

            with tf.variable_scope('posterior'):
                QZ, QW = map(normal, ['Z', 'W'])

            KL.update({Z: QZ, W: QW})

            self.data = tf.placeholder(self.dtype, shape)
            if test_data:
                self.test_data = tf.placeholder(self.dtype)
            self.data_mean = tf.placeholder(self.dtype, (shape[0], 1))
            if self.model['mu'] == 'none':
                m = self.param_init('mu', 'posterior', 'loc') # 'posterior' variables are by default trainable
                self.variables.update({'mu': m})
            elif self.model['mu'] == 'full':
                with tf.variable_scope('prior'):
                    m = normal('mu')
                with tf.variable_scope('posterior'):
                    qm = normal('mu')
                KL.update({m: qm})
                self.variables.update({'mu': qm.mean()})

            if self.model['tau'] == 'none':
                tau = self.param_init('tau', 'posterior', 'scale')
                self.variables.update({'tau': tau})
            elif self.model['tau'] == 'full':
                s = ed.models.Gamma(1e-5, 1e-5, name='prior/tau')
                qs = self.lognormal('tau')
                tau = s ** -.5
                KL.update({s: qs})
                self.variables.update({'tau': qs.mean()})

            i = tf.is_finite(self.data)
            self.data_gathered = tf.boolean_mask(self.data, i)
            x = tf.boolean_mask(tf.matmul(W, Z, transpose_b=True) + m, i)
            self.data_model = ed.models.Normal(x, tau * tf.ones(tf.shape(x)))

            self.inference = ed.KLqp(KL, data={self.data_model: self.data_gathered})

            # this comes from edward source (class VariationalInference)
            # this way, edward automatically writes out my own summaries
            summary_key = 'summaries_{}'.format(id(self.inference))

            # train_loss is not instrumental in the procedure, I compute it solely to write it out to tensorboard
            with tf.variable_scope('losses'):
                xm = tf.add(tf.matmul(QW.mean(), QZ.mean(), transpose_b=True), m, name='x')
                train_loss = tf.losses.mean_squared_error(self.data_gathered, tf.boolean_mask(xm, i))
                tf.summary.scalar('train_loss', train_loss, collections=[summary_key])
                if test_data:
                    test_loss = tf.losses.mean_squared_error(
                        tf.boolean_mask(self.test_data, i), tf.boolean_mask(xm, i))
                    tf.summary.scalar('test_loss', test_loss, collections=[summary_key])
                    self.variables.update({'test_loss': test_loss})

            self.variables.update({'x': xm, 'W': QW.mean(), 'Z': QZ.mean(), 'train_loss': train_loss})
            self.inference.initialize(n_samples=10, logdir=self.logdir)


    def run(self, data, n_iter, open_session=True, convergence_test='train_loss', test_data=None):
        """Run the actual inference after the graph has been constructed in the :class:`probPCA` init call.

        :param data: The data as a :class:`~numpy.ma.core.MaskedArray` in the shape (D, N) (see `Conventions`_)
        :param n_iter: The maximum number of iterations to run.
        :param open_session: Whether to open a `tf.InteractiveSession` (if ``False``, an interactive session with the instance's :attr:`~PPCA.graph` needs to be open).
        :param convergence_test: Which type of loss to use to test for convergence. Currently I take the StDev of the last 100 iterations of the loss function:

            * ``None`` - the training is run exactly the ``n_iter`` loops given as argument to :meth:`~probPCA.run`
            * ``train_loss`` - :attr:`train_loss` is used, the training error w.r.t. to the data passed.
            * ``elbo``, Edward_'s built-in loss is used (I think ELBO).

        """

        # NOTE: tf.InteractiveSession is the same as tf.Session except it makes the session the default session
        # Edward unfortunately seems to only use the default session, so we need to work with that.
        session = tf.InteractiveSession(graph=self.graph) if open_session is True else tf.get_default_session()

        # if this is a repeated run, replace edward's FileWriter to write to a new directory
        try:
            t = self.inference.t.eval(session)
        except tf.errors.FailedPreconditionError:
            pass
        else:
            self.logsubdir = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        feed_dict = {self.data: data, self.data_mean: np.nanmean(data, 1, keepdims=True)}
        if test_data is not None:
            feed_dict[self.test_data] = test_data
        tf.global_variables_initializer().run(feed_dict)

        # hack to use the progbar that edward allocates anyway, without giving n_iter to inference.initialize()
        self.inference.progbar.target = n_iter

        # for computing StDev of last 100 loss values as convergence criterion
        deque = np.empty(100)

        self.n_iter = n_iter
        start_time = time.time()

        if convergence_test == None:
            for i in range(n_iter):
                out = self.inference.update(feed_dict)
                self.inference.print_progress(out)
        elif convergence_test == 'train_loss':
            thresh = 1e-3
            loss = self.variables['train_loss']
            for i in range(n_iter):
                out = self.inference.update(feed_dict)
                self.inference.print_progress(out)
                j = i % 100
                deque[j] = loss.eval(feed_dict)
                if (j == 99) and (deque.std() < thresh):
                    self.n_iter = i + 1
                    break
        elif convergence_test == 'elbo':
            thresh = 40
            for i in range(n_iter):
                out = self.inference.update(feed_dict)
                self.inference.print_progress(out)
                j = i % 100
                deque[j] = out['loss']
                if (j == 99) and (deque.std() < thresh):
                    self.n_iter = i + 1
                    break

        print('\nexecution time: {}\n'.format(time.time() -  start_time))
        self.inference.finalize() # this just closes the FileWriter
        self.variables.update({'loss': self.inference.loss})
        self.convergence_test = convergence_test
        for k, v in self.variables.items():
            try:
                self.__dict__.update({k: v.eval()})
            except tf.errors.InvalidArgumentError:
                self.__dict__.update({k: v.eval(feed_dict)})

        # The test sequences aren't crashing anymore now, so maybe this was the relevant missing piece
        if open_session:
            session.close()
        return self


class vbPCA(PCA):
    """BayesPy_-based Bayesian PCA."""
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
