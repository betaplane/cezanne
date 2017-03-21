#!/usr/bin/env python
import pandas as pd
import numpy as np

S = pd.HDFStore('../../data/hydro/data.h5')
el = S['data']['Elqui']

amax = el['A'].groupby(el.index.year).apply(np.argmax)
dmax = el['D'].groupby(el.index.year).apply(np.argmax)

em = el.rolling(3).mean()
d = pd.concat((em['A'][em.index.month==7], em['D'][em.index.month==12]), 1)
