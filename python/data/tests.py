"""
Tests
-----

To run tests with mpi4py (the import form is necessary because of the config values only accessibly via the parent package :mod:`python.data`)::

    mpiexec -n 1 python -c "from data import tests; tests.run_tests()"
    # or
    mpiexec -n python -m unittest data.tests.TestField

If using :mod:`condor`, the code files need to be downloaded for the tests to work correctly (because of relative imports in the :mod:`~python.data` package)::

    mpiexec -n 1 python -c "import condor; condor.enable_sshfs_import(..., download=True); from data import tests; tests.run_tests()"

"""
import unittest, re
import xarray as xr
import numpy as np
import sys, os
from mpi4py import MPI
from importlib import import_module
from . import config, WRF, interpolate


# I can't get the xarray.testing method of the same name to work (fails due to timestamps)
class WRFTests(unittest.TestCase):
    interpolator = 'bilinear'
    n_proc = 3

    @classmethod
    def tearDownClass(cls):
        os.remove('out.nc')
        if hasattr(cls, 'data'):
            cls.data.close()

    @staticmethod
    def run_concat(interpolator):
        comm = MPI.Comm.Get_parent()
        kwargs = None
        kwargs = comm.bcast(kwargs, root=0)
        cc = WRF.Concatenator('d03', interpolator=interpolator)
        if cc.rank == 0:
            cc.files.dirs = [d for d in cc.files.dirs if re.search('c01_2016120[1-3]', d)]
        cc.concat(**kwargs)
        comm.barrier()
        comm.Disconnect()

    def setUp(self):
        self.comm =  MPI.COMM_SELF.Spawn(sys.executable,
            ['-c', 'from data import tests;tests.WRFTests.run_concat("{}")'.format(self.interpolator)],
                                         maxprocs=self.n_proc)

    def tearDown(self):
        self.comm.Disconnect()
        self.data.close()

    def run_util(self, **kwargs):
        self.comm.bcast(kwargs, root=MPI.ROOT)
        self.comm.barrier()
        self.data = xr.open_dataset('out.nc')

class T2all(WRFTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test = xr.open_dataset(config['tests']['T2_all'])

    def test_all_whole_domain(self):
        tr = ('start', 'Time', 'south_north', 'west_east')
        self.run_util(variables='T2')
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_all_interpolated(self):
        self.run_util(variables='T2', interpolate=True)
        np.testing.assert_allclose(
            self.data['T2'].transpose('start', 'Time', 'station'),
            self.test['interp'].transpose('start', 'Time', 'station'), rtol=1e-3)

class T2lead1(WRFTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test = xr.open_dataset(config['tests']['T2_lead1'])

    def test_lead_day_whole_domain(self):
        tr = ('Time', 'south_north', 'west_east')
        self.run_util(variables='T2', lead_day=1)
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_lead_day_interpolated(self):
        tr = ('Time', 'station')
        self.run_util(variables='T2', lead_day=1, interpolate=True)
        np.testing.assert_allclose(
            self.data['T2'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-4)

class Tall(WRFTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test = xr.open_dataset(config['tests']['T_all'])

    def test_all_whole_domain(self):
        tr = ('start', 'Time', 'bottom_top', 'south_north', 'west_east')
        self.run_util(variables='T')
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_all_interpolated(self):
        tr = ('start', 'Time', 'bottom_top', 'station')
        self.run_util(variables='T', interpolate=True)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-3)

class Tlead1(WRFTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test = xr.open_dataset(config['tests']['T_lead1'])

    def test_all_whole_domain(self):
        tr = ('Time', 'bottom_top', 'south_north', 'west_east')
        self.run_util(variables='T', lead_day=1)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['field'].transpose(*tr))

    def test_all_interpolated(self):
        tr = ('Time', 'bottom_top', 'station')
        self.run_util(variables='T', lead_day=1, interpolate=True)
        np.testing.assert_allclose(
            self.data['T'].transpose(*tr),
            self.test['interp'].transpose(*tr), rtol=1e-3)

def run_tests():
    WRFTests.interpolator = 'scipy'
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2all))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2lead1))
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Tall))
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Tlead1))
    # suite.addTest(Tlead1('test_all_interpolated'))
    runner = unittest.TextTestRunner()
    runner.run(suite)

