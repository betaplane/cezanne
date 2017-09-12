#!/usr/bin/env python
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from bayespy.nodes import GaussianARD, Gamma, SumMultiply, Add, GaussianGamma
from bayespy.inference import VB
from bayespy.inference.vmp.transformations import RotateGaussianARD, RotationOptimizer
from timeit import default_timer as timer

# https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html

ds = xr.open_mfdataset('../../data/SST/*.nc')
lm = xr.open_dataarray('../../data/SST/lsmask.nc')
sst = ds['sst'].resample('MS', 'time', how='mean')
y, m = [x.sel(lon=slice(145, 295), lat=slice(10, -35)) for x in [sst, lm]]

class pca_base(object):
    def __init__(self, Y, mask, mean=True, std=True):
        s = lambda x: x.stack(x = ('lat', 'lon'))
        if mean:
            Y = Y - Y.mean('time')
        if std:
            Y = Y / Y.std('time')
        self.Y = s(Y).sel(x = s(mask).values.astype(bool).flatten()).transpose('x', 'time')
        self.d, self.n = self.Y.shape

    def plot(self, d=10):
        fig, axs = plt.subplots(d, 2)
        W = self.W.unstack('x')
        for i, d in enumerate((self.W ** 2).sum('x').argsort().values[::-1]):
            axs[i, 0].plot(self.X.isel(d = d))
            p = axs[i, 1].imshow(W.isel(d = d))
            plt.colorbar(p, ax = axs[i, 1])
        plt.show()

    def err(self, d):
        X = self.X.sel(d = slice(None, d)).transpose('time', 'd').values
        W = self.W.sel(d = slice(None, d)).transpose('d', 'x').values
        self.recon = X.dot(W)
        return ((self.Y.transpose('time', 'x') - self.recon) ** 2).mean().compute()

    def sort(self):
        i = np.argsort((self.W ** 2).sum('x'))[::-1]
        self.W = self.W.isel(d = i)
        self.X = self.X.isel(d = i)

    def rotate(self):
        X = self.X.transpose('d', 'time')
        W = self.W.transpose('x', 'd')
        v, U = np.linalg.eigh(X.values.dot(X.T))
        Dx = np.diag(v ** .5)
        v, V = np.linalg.eigh(Dx.dot(U.T).dot(W.T).dot(W).dot(U).dot(Dx))
        self.W = xr.DataArray(W.values.dot(U).dot(Dx).dot(V), coords = W.coords)
        self.X = xr.DataArray(V.T.dot(np.diag(v ** -.5)).dot(U.T).dot(X), coords = X.coords)

class dpca(pca_base):
    def __call__(self, d=None):
        w, v = np.linalg.eigh(np.cov(self.Y)) # eigh for symmetric / hermitian matrices
        d = v.shape[1] if d is None else d
        i = np.argsort(w)[::-1][:d]
        self.W = xr.DataArray(v[:, i], coords = [('x', self.Y.indexes['x']), ('d', np.arange(d))])
        self.X = self.Y.dot(self.W)


# http://bayespy.org/en/latest/examples/pca.html

# Ilin, Alexander, and Tapani Raiko. 2010. “Practical Approaches to Principal Component Analysis in the Presence of Missing Values.” Journal of Machine Learning Research 11 (Jul): 1957–2000.

# NOTE: PPCA (probabilistic PCA) would not put a prior on PC means and variances, amounting to less regularization - can be solved with standard EM.

# NOTE: to be in line with (Ilin and Raiko 2010), one would probably have to include an explicit model for the bias vector ('m'), which in the VBPCA *or* PPCA formulation with *missing* data is *not* equivalent to the bias of the observations.

class vbpca(pca_base):
    def __call__(self, d=None):
        # dimensionality of principal subspace
        d = len(self.Y.x) if d is None else d

        # principal components ('latent variables')
        X = GaussianARD(0, 1, plates=(1, len(self.Y.time)), shape=(d, ))

        # "plates" share a distribution, whereas "shape" refers to independent nodes

        # ARD (automatic relevance determination) refers to putting a shared prior on the variance of the PC vectors (here: alpha). This amounts to automatic selection of the regularization paramter in ridge regression. If evidence of a component's relevance is weak, its variance tends to zero. See Ilin and Raiko (2010, sec 3.3); Hastie, sec. 10.9.1.
        # http://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard

        # prior over precision of loading matrix
        # NOTE: if this was a non-constant (learned) distribution, it would have to be
        # conjugate to the child node (i.e. Gaussian in this case)
        alpha = Gamma(1e-5, 1e-5, plates=(d, ))

        # loadings matrix
        W = GaussianARD(0, alpha, plates=(len(self.Y.x), 1), shape=(d, ))

        # ma = GaussianGamma(0, 1e-5, 1e-5, 1e-5, plates=(len(self.Y.x), 1))
        # mu = Gaussian(25, 1e-5, plates=(len(self.Y.x), ))
        # M = GaussianARD(ma, 1, plates=(len(self.Y.x), 1))

        # observations
        F = SumMultiply('d,d->', X, W)
        # G = Add(F, M)
        tau = Gamma(1e-5, 1e-5) # this is the observation noise
        Y = GaussianARD(F, tau)
        Y.observe(self.Y)

        # inference
        Q = VB(Y, X, W, alpha, tau)
        W.initialize_from_random()

        # rotations at each iteration to speed up convergence
        rot_X = RotateGaussianARD(X)
        rot_C = RotateGaussianARD(W, alpha)
        R = RotationOptimizer(rot_X, rot_C, d)
        Q.set_callback(R.rotate)

        Q.update(repeat=1000)

        # get first moments of posterior (means)
        w = W.get_moments()[0].squeeze()
        x = X.get_moments()[0].squeeze()
        # m = M.get_moments()[0].squeeze()
        i = np.argsort(np.diag(w.T.dot(w)))[::-1]
        self.W = xr.DataArray(w[:, i], coords = [('x', self.Y.indexes['x']), ('d', np.arange(d))])
        self.X = xr.DataArray(x[:, i], coords = [('time', self.Y.indexes['time']), ('d', np.arange(d))])
        # self.m = xr.DataArray(m, coords = [('x', self.Y.indexes['x'])])
        self.tau = tau.get_moments()[0].squeeze()

# http://pymc-devs.github.io/pymc3/index.html
class mcpca(pca_base):
    def construct(self, d=None):
        import pymc3 as pm
        d = len(self.Y.x) if d is None else d
        self.model = pm.Model()

        with self.model:
            X = pm.Normal('X', mu = 0, tau = 1, shape = (d, len(self.Y.time)))
            alpha = pm.Gamma('alpha', alpha = 1e-5, beta = 1e-5, shape = d)
            W = pm.Normal('W', mu = 0, tau = alpha, shape = (len(self.Y.x), d),
                          testval = np.random.randn(len(self.Y.x), d))
            F = pm.math.dot(W, X)
            tau = pm.Gamma('tau', alpha = 1e-5, beta = 1e-5)
            Y = pm.Normal('Y', mu = F, tau = tau, observed = self.Y)

            # self.trace = pm.sample(1000, tune = 500)
            # self.map = pm.find_MAP()

    def vb(self):
        self.fit = pm.fit(model = self.model, method = 'advi')
        s = self.fit.sample(1000)
        w = s.get_values('W')
        x = s.get_values('X')
        self.extract(w, x)

    def mcmc(self):
        self.trace = pm.sample(1000, tune = 500)
        w = self.trace['W'].mean(0)
        x = self.trace['X'].mean(0)
        self.extract(w, x)

    def extract(self, w, x):
        d = np.arange(w.shape[-1])
        i = np.argsort(np.diag(w.T.dot(w)))[::-1]
        self.W = xr.DataArray(w[:, i], coords=[('x', self.Y.indexes['x']), ('d', d)])
        self.X = xr.DataArray(x[i, :], coords=[('d', d), ('time', self.Y.indexes['time'])])


import pystan as ps
import pandas as pd

# http://pystan.readthedocs.io/en/latest/index.html
class stan(pca_base):
    def __init__(self, *args, model=None):
        super(stan, self).__init__(*args)
        if model is not None:
            self.model = model

    def read_csv(self, filename=None, d=None, vars={'X': 'X', 'W': 'W'}):
        import MyStan
        d = self.d if d is None else d
        filename = self.filename if filename is None else filename
        x = MyStan.read(filename, vars['X']).mean(0)
        w = MyStan.read(filename, vars['W']).mean(0)
        self.reshape(w, x, d)

    def reshape(self, w, x, d):
        self.W = xr.DataArray(w.reshape((d, -1)),
                         coords=[('d', np.arange(d)), ('x', self.Y.indexes['x'])])
        self.X = xr.DataArray(x.reshape((-1, d)),
                         coords=[('time', self.Y.indexes['time']), ('d', np.arange(d))])

    def space(self, x):
        return xr.DataArray(x, coords=[('x', self.Y.indexes['x'])]).unstack('x')

    def read(self, filename=None, d=None, vars={'X': 'X', 'Y': 'Y'}):
        d = self.d if d is None else d
        filename = self.filename if filename is None else filename
        with open(filename) as f:
            while True:
                l = next(f)
                if l[0] == '#':
                    continue
                else:
                    break
            self.columns = l.split(',')

        cx = [i for i, j in enumerate(self.columns) if j[0] == vars['X']]
        cw = [i for i, j in enumerate(self.columns) if j[0] == vars['W']]

        w = pd.read_csv(filename, skiprows=7, usecols=cw, header=None).mean(0).values
        x = pd.read_csv(filename, skiprows=7, usecols=cx, header=None).mean(0).values 
        self.W = xr.DataArray(w.reshape((d, -1)),
                         coords=[('d', np.arange(d)), ('x', self.Y.indexes['x'])])
        self.X = xr.DataArray(x.reshape((-1, d)),
                         coords=[('time', self.Y.indexes['time']), ('d', np.arange(d))])

    def to_netcdf(self, name):
        ds = xr.Dataset({'W': self.W.unstack('x'), 'X': self.X})
        ds.to_netcdf(name)

    def construct(self, code):
        self.model = ps.StanModel(model_code = code)

    @property
    def data(self):
        Y = self.Y.values.flatten('F') # apparently stan does it the Fortran way round
        i =  np.where(~np.isnan(Y))[0]
        s = self.Y.shape
        # remember, stan is 1-indexed
        return {'x': s[0], 'n': s[1], 'Y': Y[i], 'obs': i + 1, 'n_obs': len(i)}

    def vb(self, d, sample_file):
        data = self.data
        data['d'] = d
        start = timer()
        self.fit = self.model.vb(data = data, sample_file = sample_file)
        print(timer() - start)
        self.filename = self.fit['args']['sample_file'].decode()

code_basic = """
data {
    int<lower=1> n;
    int<lower=1> x;
    int<lower=1> d;
    int<lower=1> n_obs;
    vector[n_obs] Y;
    int<lower=1, upper=n*x> obs[n_obs];
}
parameters {
    matrix[x, d] W;
    matrix[d, n] X;
    vector<lower=0>[d] alpha;
    real<lower=0> tau;
}
model {
    alpha ~ gamma(1e-5, 1e-5);
    tau ~ gamma(1e-5, 1e-5);
    to_vector(X) ~ normal(0, 1);
    for (i in 1:x)
        W[i] ~ normal(0, alpha);
    Y ~ normal(to_vector(W * X)[obs], tau);
}
        """
# sm = ps.StanModel(model_code=code_basic)

stan_mean = """
data {
    int<lower=1> n;
    int<lower=1> x;
    int<lower=1> d;
    int<lower=1> n_obs;
    vector[n_obs] Y;
    int<lower=1, upper=n*x> obs[n_obs];
}
parameters {
    matrix[x, d] W;
    matrix[d, n] X;
    vector[x] m;
    vector<lower=0>[x] vm;
 //   vector[x] mu;
    vector<lower=0>[d] alpha;
    real<lower=0> tau;
}
model {
    alpha ~ gamma(1e-5, 1e-5);
    tau ~ gamma(1e-5, 1e-5);
    vm ~ gamma(1e-5, 1e-5);
//    mu ~ normal(0, 1e-5);
    m ~ normal(0, vm);
    to_vector(X) ~ normal(0, 1);
    for (i in 1:x)
        W[i] ~ normal(0, alpha);
    Y ~ normal(to_vector(W * X + rep_matrix(m, n))[obs], tau);
}
        """
# sm = ps.StanModel(model_code=stan_mean)

code_rotated = """
data {
    int<lower=1> n;
    int<lower=1> x;
    int<lower=1> d;
    matrix[x, n] Y;
}
parameters {
    matrix[x, d] W;
    matrix[d, n] X;
    vector<lower=0>[d] alpha;
    real<lower=0> tau;
}
transformed parameters {
    cov_matrix[d] xx = tcrossprod(X);
    vector<lower=0>[d] l = sqrt(eigenvalues_sym(xx));
    matrix[d, d] U = eigenvectors_sym(xx);
    cov_matrix[d] Dx = diag_matrix(l);
    cov_matrix[d] Dx_inv = diag_matrix(inv(l));
    matrix[d, d] V = eigenvectors_sym(Dx * U' * crossprod(W) * U * Dx);
    matrix[x, d] Wo = W * U * Dx * V;
    matrix[d, n] Xo = V' * Dx_inv * U' * X;
}
model {
    alpha ~ gamma(1e-5, 1e-5);
    tau ~ gamma(1e-5, 1e-5);
    to_vector(X) ~ normal(0, 1);
    for (i in 1:x)
        W[i] ~ normal(0, alpha);
    to_vector(Y) ~ normal(to_vector(Wo * Xo), tau);
}
        """
# sm = ps.StanModel(model_code=code)


import edward as ed
import tensorflow as tf

# https://gist.github.com/pwl/2f3c3e240b477eac9a37b06791b2a659

class edpca(pca_base):
    def __init__(self, *args, k=None):
        super(edpca, self).__init__(*args)
        if k is None:
            k = self.d

        Y = self.Y.values.flatten()
        i = np.where(~np.isnan(Y))[0]

        mm = tf.Variable(tf.random_normal((self.d, 1)) + self.Y.mean('time').values.reshape((-1, 1)))
        vm = tf.Variable(1.0)
        m = ed.models.Normal(mm, vm * tf.ones((self.d, 1)))
        s = ed.models.Gamma(1e-5, 1e-5)
        W = ed.models.Normal(tf.zeros((self.d, k)), tf.ones((self.d, k)))
        Z = ed.models.Normal(tf.zeros((k, self.n)), tf.ones((k, self.n)))

        self.qw = ed.models.Normal(tf.Variable(tf.random_normal((self.d, k))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((self.d, k)))))
        self.qz = ed.models.Normal(tf.Variable(tf.random_normal((k, self.n))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((k, self.n)))))
        self.s = ed.models.TransformedDistribution(
            ed.models.NormalWithSoftplusScale(tf.Variable(0.0), tf.Variable(1.0)),
            bijector = tf.contrib.distributions.bijectors.Exp())
        self.m = ed.models.Normal(mm, tf.nn.softplus(tf.random_normal((self.d, 1))))

        # x = tf.gather(tf.reshape(tf.matmul(W, Z) + m, [-1]), i)
        # X = ed.models.Normal(x, tf.ones(len(i)) * tf.pow(s, tf.constant(-0.5)))
        # self.inference = ed.KLqp({W: self.qw, Z: self.qz, s: self.s, m: self.m}, data={X: Y[i]})

        X = ed.models.Normal(tf.matmul(W, Z) + m, tf.ones((self.d, self.n)) * tf.pow(s, tf.constant(-0.5)))
        self.inference = ed.KLqp({W: self.qw, Z: self.qz, s: self.s, m: self.m}, data={X: self.Y.values})

    def __call__(self, n_iter):
        self.out = self.inference.run(n_iter=n_iter)
        self.extract()



    def extract(self, k=10):
        s = ed.get_session()
        self.W = xr.DataArray(s.run(self.qw.mean()),
                              coords=[('x', self.Y.indexes['x']), ('d', np.arange(k))])
        self.X = xr.DataArray(s.run(self.qz.mean()),
                              coords=[('d', np.arange(k)), ('time', self.Y.indexes['time'])])


class edvar(edpca):
    def __init__(self, *args, k=None):
        pca_base.__init__(self, *args)
        if k is None:
            k = self.d

        self.W = ed.models.Normal(tf.zeros((self.d, k)), tf.ones((self.d, k)))
        self.Z = ed.models.Normal(tf.zeros((k, self.n)), tf.ones((k, self.n)))

        X = ed.models.Normal(tf.matmul(W, Z), tf.ones((self.d, self.n)))

        self.qw = ed.models.Normal(tf.Variable(tf.random_normal((self.d, k))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((self.d, k)))))
        self.qz = ed.models.Normal(tf.Variable(tf.random_normal((k, self.n))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((k, self.n)))))

        self.inference = ed.KLqp({}, data={X: self.Y.values})

    def inference(self, n_iter):
        self.inference.initialize()
        tf.global_variables_initializer().run()
        for n in range(n_iter):
            upd = self.inference.update({self.W: self.qw, self.Z: self.qz})
            Z = self.qz.mean()
            W = self.qw.mean()
            u, U = tf.self_adjoint_eig(tf.matmul(Z, transpose_b=True))
            Dx = tf.diag(tf.pow(u, .5))
            Ux = tf.matmul(U, Dx)
            v, V = tf.self_adjoint_eig(
                tf.matmul(
                    tf.matmul(Ux,
                              tf.matmul(W, W, transpose_a=True),
                              transpose_a=True),
                    Ux)
            )
            Wu = tf.matmul(tf.matmul(W, Ux), V)
            Zu = tf.matmul(
                tf.matmul(
                    tf.matmul(V, tf.diag(tf.pow(u, -.5)), transpose_a=True),
                    U, transpose_b=True), Z)
            if n%10==0:
                self.inference.print_progress(upd)
        self.inference.finalize()

        
class edmpca(pca_base):
    def __call__(self, k=None, n_iter=500):
        if k is None:
            k = self.d

        Y = self.Y.values.flatten()
        i = np.where(~np.isnan(Y))[0]

        W = ed.models.Normal(tf.zeros((self.d, k)), tf.ones((self.d, k)))
        Z = ed.models.Normal(tf.zeros((k, self.n)), tf.ones((k, self.n)))

        x = tf.gather(tf.reshape(tf.matmul(W, Z), [-1]), i)
        X = ed.models.Normal(x, tf.ones(len(i)))

        self.qw = ed.models.Normal(tf.Variable(tf.random_normal((self.d, k))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((self.d, k)))))
        self.qz = ed.models.Normal(tf.Variable(tf.random_normal((k, self.n))),
                              tf.nn.softplus(tf.Variable(tf.random_normal((k, self.n)))))

        # Xobs = tf.gather(tf.reshape(X, [-1]), i)
        # self.x = ed.models.NormalWithSoftplusScale(tf.Variable(tf.random_normal((self.d, self.n))),
        #                                            tf.Variable(tf.random_normal((self.d, self.n))))

        # self.inference = ed.KLqp({W: self.qw, Z: self.qz, X: self.x}, data={Xobs: Y[i]})
        self.inference = ed.KLqp({W: self.qw, Z: self.qz}, data={X: Y[i]})
        self.out = self.inference.run(n_iter=n_iter)

        s = ed.get_session()
        self.W = xr.DataArray(s.run(self.qw.mean()),
                              coords=[('x', self.Y.indexes['x']), ('d', np.arange(k))])
        self.X = xr.DataArray(s.run(self.qz.mean()),
                              coords=[('d', np.arange(k)), ('time', self.Y.indexes['time'])])
