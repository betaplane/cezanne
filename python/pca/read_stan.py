#!/usr/bin/env python
import pandas as pd

def read(filename, var):
    with open(filename) as f:
        while True:
            l = next(f)
            if l[0] == '#':
                continue
            else:
                break
        c = [i for i, j in enumerate(l.split(',') if j[0] == var]

    return pd.read_csv(filename, skiprows=7, usecols=c, header=None)
