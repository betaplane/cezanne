#!/usr/bin/env python
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from bayespy.nodes import GaussianARD, Gamma, SumMultiply
from bayespy.inference import VB
from bayespy.inference.vmp.transformations import RotateGaussianARD, RotationOptimizer

# https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html

ds = xr.open_mfdataset('../../data/SST/*.nc')
lm = xr.open_dataarray('../../data/SST/lsmask.nc')
sst = ds['sst'].resample('MS', 'time', how='mean')
y, m = [x.sel(lon=slice(145, 295), lat=slice(10, -35)) for x in [sst, lm]]

class pca_base(object):
    def __init__(self, Y, mask):
        s = lambda x: x.stack(x = ('lat', 'lon'))
        self.ym = Y.mean('time')
        y = ((Y - self.ym) / Y.std('time'))
        self.Y = s(y).sel(x = s(mask).values.astype(bool).flatten()).transpose('x', 'time')

    def plot(self, d=10):
        fig, axs = plt.subplots(d, 2)
        W = self.W.unstack('x')
        for i in range(d):
            axs[i, 0].plot(self.X.sel(d = i))
            p = axs[i, 1].imshow(W.sel(d = i))
            plt.colorbar(p, ax = axs[i, 1])
        plt.show()

    def err(self, d):
        self.recon = self.X.sel(d=slice(None, d)).values.dot(self.W.sel(d=slice(None, d)).transpose('d', 'x'))
        return ((self.Y.transpose('time', 'x') - self.recon) ** 2).mean().compute()

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

        # observations
        F = SumMultiply('d,d->', X, W)
        tau = Gamma(1e-5, 1e-5) # but why do we need this?
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
        i = np.argsort(np.diag(w.T.dot(w)))[::-1]
        self.W = xr.DataArray(w[:, i], coords = [('x', self.Y.indexes['x']), ('d', np.arange(d))])
        self.X = xr.DataArray(x[:, i], coords = [('time', self.Y.indexes['time']), ('d', np.arange(d))])

import pymc3 as pm

class mcpca(pca_base):
    def __call__(self, d=None):
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
            self.map = pm.find_MAP()
