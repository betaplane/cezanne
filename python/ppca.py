#!/usr/bin/env python
import xarray as xr
import numpy as np
from bayespy.nodes import GaussianARD, Gamma, SumMultiply
from bayespy.inference import VB
from bayespy.inference.vmp.transformations import RotateGaussianARD RotationOptimizer

# https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html

ds = xr.open_mfdataset('../SST/*.nc')
lm = xr.open_dataarray('../SST/lsmask.nc')

def select(sst, lm, lon, lat):
    x = sst.sel(lon = lon, lat = lat).values
    m = lm.sel(lon = lon, lat = lat).values
    i = np.where(m.flatten())[0]
    return x.reshape((x.shape[0], -1))[:, i], i, x.shape[1:]

def eof(v, j, shape):
    x = np.ones(np.prod(shape)) * np.nan
    x[j] = v
    return x.reshape(shape)

y, j, s = select(ds['sst'], lm, slice(145, 295), slice(10, -35))

c = np.cov(y.T)
w, v = np.linalg.eigh(c)
i = np.argsort(w)

# http://bayespy.org/en/latest/examples/pca.html

m = y.mean(0)
y = (y - m) / y.std(0)

D = 10
X = GaussianARD(0, 1, plates=(1, y.shape[0]), shape=(D, ))
alpha = Gamma(1e-5, 1e-5, plates=(D, ))
C = GaussianARD(0, alpha, plates=(y.shape[1], 1), shape=(D,))
F = SumMultiply('d,d->', X, C)
tau = Gamma(1e-5, 1e-5)
Y = GaussianARD(F, tau)
Y.observe(y.T)
Q = VB(Y, X, C, alpha, tau)
C.initialize_from_random()

rot_X = RotateGaussianARD(X)
rot_C = RotateGaussianARD(C, alpha)

R = RotationOptimizer(rot_X, rot_C, D)
Q.set_callback(R.rotate)

c = C.get_moments()[0] # first moment?
