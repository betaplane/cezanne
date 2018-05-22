import core, tests
import numpy as np
import xarray as xr
import pandas as pd

x, mu, Z, W, n = [], [], [], [], []

d = tests.Data().real(ta_c=1)

def cc(a, c, exp):
    b = a.expand_dims('exp')
    b['exp'] = ('exp', [exp])
    c.append(b)
    return [xr.concat(c, 'exp')]


for m in np.arange(0, 1, .1):
    for s in range(10):
        for b in [10, 20]:
            d.missing(m, b)
            for K in range(1, 5):
                print(m, s, b, K)
                idx = ('K', range(K))
                p = core.detPCA(blocks=b, s=s).run(d.x1, test_data=d.x, n_iter=100, K=K)
                p.critique(d, file_name='lima_det_blocks2.h5', table_name='determ/results', rotate=False)
                x = cc(xr.DataArray(p.x, coords=[d.x.index, d.x.columns]), x, p.id)
                mu = cc(xr.DataArray(p.mu.flatten(), coords=[d.x.index]), mu, p.id)
                Z = cc(xr.DataArray(p.Z, coords=[d.x.columns, idx]), Z, p.id)
                W = cc(xr.DataArray(p.W, coords=[
                    ('D', d.x.index.values), idx]), W, p.id)
                ds = xr.Dataset({'x': x[0], 'mu': mu[0], 'Z': Z[0], 'W': W[0]})

                ds.to_netcdf('lima_det_blocks2.nc')
