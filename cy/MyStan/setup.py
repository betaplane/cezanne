from distutils.core import setup, Extension
from Cython.Build import cythonize

# http://setuptools.readthedocs.io/en/latest/setuptools.html#metadata
setup(name = 'MyStan', version = '0.1',
    ext_modules = cythonize(Extension(
    'MyStan', ['MyStan.pyx'], language='c++',
    extra_compile_args = ['-std=c++11']
)))
