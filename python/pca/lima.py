import core, tests
import numpy as np
import xarray as xr
import pandas as pd

x, mu, tau, Z, W, alpha, n = [], [], [], [], [], [], []

d1 = tests.Data().real(ta_c=0)
d2 = tests.Data().real(ta_c=1)

def cc(a, c, exp):
    b = a.expand_dims('exp')
    b['exp'] = ('exp', [exp])
    c.append(b)
    return [xr.concat(c, 'exp')]


for m in np.arange(0, 1, .1):
    for d in [d1, d2]:
        d.missing(m, 20)
        for s in range(20):
            P = []
            try:
                p1 = core.probPCA(d.x.shape, logdir='lima_logs', test_data=True, seed=s)
                p1.run(d.x1, n_iter=20000, test_data=d.x)
                p1.critique(d, file_name='lima_blocks.h5', table_name='edward/results')
            except:
                pass
            else:
                P.append(p1)

            try:
                p2 = core.vbPCA(d.x1, n_iter=2000, rotate=True)
                p2.critique(d, file_name='lima_blocks.h5', table_name='bayespy/results', rotate=False)
            except:
                pass
            else:
                alpha = cc(xr.DataArray(p2.alpha, coords=[d.x.index]), alpha, p2.id)
                P.append(p2)

            if len(P) > 0:
                for i, p in enumerate(P):
                    x = cc(xr.DataArray(p.x, coords=[d.x.index, d.x.columns]), x, p.id)
                    mu = cc(xr.DataArray(p.mu.flatten(), coords=[d.x.index]), mu, p.id)
                    tau = cc(xr.DataArray(p.tau), tau, p.id)
                    Z = cc(xr.DataArray(p.Z, coords=[d.x.columns, d.x.index]), Z, p.id)
                    W = cc(xr.DataArray(p.W, coords=[
                        ('D', d.x.index.values), ('K', d.x.index.values)]), W, p.id)
                ds = xr.Dataset({'x': x[0], 'mu': mu[0], 'tau': tau[0], 'Z': Z[0], 'W': W[0], 'alpha': alpha[0]})
                ds.to_netcdf('lima_blocks.nc')

