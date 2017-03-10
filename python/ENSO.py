#!/usr/bin/env python
import pandas as pd


# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
oni = pd.read_csv('/home/arno/Downloads/oni.data',
                  delim_whitespace=True,
                  skiprows=1,
                  skipfooter=8,
                  index_col=0,
                  na_values=-99.99,
                  header=None).stack()

def stack2index(index):
    return pd.DatetimeIndex(['{}-{}'.format(*i) for i in index.tolist()])

oni.index = stack2index(oni.index)
