#!/usr/bin/env python
import pandas as pd
from shapely.geometry import Polygon, MultiPoint

d3 = xr.open_dataset('../../data/WRF/3d/geo_em.d03.nc')

# all corner points (for different staggerings) - last one encompasses the other ones
coords = np.r_[d3.corner_lons, d3.corner_lats].reshape((2, 4, -1)).transpose((1, 2, 0))
coords = np.r_['1,3', coords, coords[:,:1,:]] # fifth point = first point for polygons

# take the last corner polygon
poly = Polygon(coords.tolist()[3])

d = pd.read_csv('../../data/DGA/cr2_prDaily_2017_ghcn/cr2_prDaily_2017_ghcn.txt',
                  header=list(range(15)), index_col=0, parse_dates=True, na_values=-9999)


p = MultiPoint(d.columns.to_series().reset_index()[['longitud', 'latitud']].values)
x = d.loc[:, [poly.contains(i) for i in p]].dropna(0, 'all')
