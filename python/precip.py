#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data

# D = data.Data()

def clean(x):
    x = x.values.flatten()
    d = np.diff(x, 1)
    return np.where((d[:-1] != 0) & (d[:-1] == -d[1:]))[0] + 1

def diff(x):
    d = x.dropna().diff(1)
    d[d < 0] = np.nan
    return d


