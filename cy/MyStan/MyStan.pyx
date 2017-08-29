from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "stan_read.cpp":
     vector[vector[float]] read_var(string, string)

def read(file_name, var_name):
    return np.array(read_var(<string>file_name.encode('utf-8'), <string>var_name.encode('utf-8')))