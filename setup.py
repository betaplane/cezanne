from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules = cythonize([
        Extension('netblitz', ['netblitz.pyx'], language = 'c++',
                  include_dirs = [
                      '/usr/local/include/',
                      '/sata1_ceazalabs/arno/HPC/uvHome/miniconda3/envs/cxx/include'],
                  libraries = ['netcdf_c++4', 'blitz', 'nb'],
                  library_dirs = ['/sata1_ceazalabs/arno/HPC/uvHome/miniconda3/envs/cxx/lib',
                                  '/usr/local/lib',
                                  '.'],
                  extra_compile_args = ['-std=c++0x'],
                  )
    ])
)
