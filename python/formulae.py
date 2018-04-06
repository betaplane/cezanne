#!/usr/bin/env python
import numpy as np
import pandas as pd


def e_sat(T):
	"""
T in K, p in Pa
http://iopscience.iop.org/article/10.1088/0143-0807/33/2/295/meta
	"""
	return 610.94 * np.exp(17.625*(T-273.15)/(T-30.11))

def e(w,p):
	return w / (w + 0.622) * p

def w_sat(T,p):
	"T in K, p in Pa"
	es = e_sat(T)
	return 0.622 * es / (p-es)

def w2rh(w,T,p):
	"T in K, p in Pa"
	return 100 * w / w_sat(T,p)

def rh2w(rh,T,p):
	"T in K, p in Pa"
	return rh * w_sat(T,p) / 100

def th2T(th,p):
	"T in K, p in Pa"
	return th * (p/100000)**.286

def wind(vv_ms, dv):
    u = pd.concat((vv_ms, -np.sin(np.deg2rad(dv))), 1).product(1)
    v = pd.concat((vv_ms, -np.cos(np.deg2rad(dv))), 1).product(1)
    return u, v
