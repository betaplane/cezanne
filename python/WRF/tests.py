"""
Tests
-----

To run tests with :mod:`mpi4py` (the import form is necessary because of the config values only accessibly via the parent package :mod:`python.data`)::

    mpiexec -n 1 python -c "from data import tests; tests.runner.run(tests.scipy_suite_T2)"
    # or
    mpiexec -n python -m unittest data.tests.T2_all

If using :mod:`condor`, the code files need to be downloaded for the tests to work correctly (because of relative imports in the :mod:`~python.data` package)::

    mpiexec -n 1 python -c "import condor; condor.enable_sshfs_import(..., download=True); from data import tests; tests.runner.run(tests.scipy_suite_T2)"

The module-level variable :data:`n_proc` can be changed (default is 3)  to test if the same results are obtained with a different number or processors; two different tests suites using the two different interpolators are already being provided at the end of the file (see :data:`interpolator`)

Tests in this module should work with both versions of the WRF-Concatenator.

.. NOTE::

    The multiple inheritance of :class:`WRFTests` might be a problem in the future. Currently it seems that unittest, when run with the idiom ``python -m unittest WRF.tests``, calls :meth:`WRFTests.__init__` with the test method name as argument. If :class:`traitlets.config.Application` command line switches can be used I'm not sure at this point.

"""
import unittest, re, os
import xarray as xr
import numpy as np
import sys, os
from importlib import import_module
from importlib.util import find_spec
from data import interpolate
from traitlets.config import Application
from traitlets import Unicode

if find_spec('mpi4py'):
    MPI = import_module('mpi4py.MPI')
    WRF = import_module('.WRF_mpi', 'WRF')
else:
    WRF = import_module('.WRF_threaded', 'WRF')

interpolator = 'scipy'
"""which interpolator to use (see :mod:`.interpolate`)"""

n_proc = 3
"""Number of processors to use for the test"""


# xarray.testing methods compare dimensions etc too, which we don't want here
class WRFTests(Application, unittest.TestCase):
    file_name = Unicode().tag(config=True)
    def __init__(self, *args, **kwargs):
        Application.__init__(self)
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))

    @classmethod
    def setUpClass(cls):
        if 'mpi4py.MPI' not in sys.modules:
            cls.cc = WRF.Concatenator('d03', interpolator=interpolator)
            cls.cc.dirs = [d for d in cls.cc.dirs if re.search('c01_2016120[1-3]', d)]

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile('out.nc'):
            os.remove('out.nc')
        if hasattr(cls, 'data'):
            cls.data.close()

    @classmethod
    def run_concat(cls, interpolator):
        comm = MPI.Comm.Get_parent()
        kwargs = None
        kwargs = comm.bcast(kwargs, root=0)
        cc = WRF.Concatenator('d03', interpolator=interpolator)
        if cc.rank == 0:
            cc.dirs = [d for d in cc.dirs if re.search('c01_2016120[1-3]', d)]
        cc.concat(**kwargs)
        comm.barrier()
        comm.Disconnect()

    def setUp(self):
        self.test = xr.open_dataset(self.file_name)
        if not hasattr(self, 'cc'):
            self.comm =  MPI.COMM_SELF.Spawn(sys.executable,
                ['-c', 'from data import tests;tests.WRFTests.run_concat("{}")'.format(interpolator)],
                                         maxprocs=n_proc)

    def tearDown(self):
        if hasattr(self, 'comm'):
            self.comm.Disconnect()
        self.data.close()

    def run_util(self, **kwargs):
        if hasattr(self, 'comm'):
            self.comm.bcast(kwargs, root=MPI.ROOT)
            self.comm.barrier()
            self.data = xr.open_dataset('out.nc')
        else:
            self.cc.concat(**kwargs)
            self.data = self.cc.data

class T2_all(WRFTests):
    def test_whole_domain(self):
        tr = ('start', 'Time', 'south_north', 'west_east')
        self.run_util(variables='T2')
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_interpolated(self):
        self.run_util(variables='T2', interpolate=True)
        np.testing.assert_allclose(
            self.data['T2'].transpose('start', 'Time', 'station'),
            self.test['interp'].transpose('start', 'Time', 'station'), rtol=1e-3)

class T2_lead1(WRFTests):
    def test_whole_domain(self):
        tr = ('Time', 'south_north', 'west_east')
        self.run_util(variables='T2', lead_day=1)
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_interpolated(self):
        tr = ('Time', 'station')
        self.run_util(variables='T2', lead_day=1, interpolate=True)
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-4)

class T_all(WRFTests):
    def test_whole_domain(self):
        tr = ('start', 'Time', 'bottom_top', 'south_north', 'west_east')
        self.run_util(variables='T')
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_interpolated(self):
        tr = ('start', 'Time', 'bottom_top', 'station')
        self.run_util(variables='T', interpolate=True)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-3)

class T_lead1(WRFTests):
    def test_whole_domain(self):
        tr = ('Time', 'bottom_top', 'south_north', 'west_east')
        self.run_util(variables='T', lead_day=1)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_interpolated(self):
        tr = ('Time', 'bottom_top', 'station')
        self.run_util(variables='T', lead_day=1, interpolate=True)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-3)


runner = unittest.TextTestRunner()

# n_proc = 2
# interpolator = 'bilinear'

T2_suite = unittest.TestSuite()
T2_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2_all))
T2_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2_lead1))

T_suite = unittest.TestSuite()
T_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T_all))
T_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T_lead1))
