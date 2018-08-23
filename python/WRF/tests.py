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
__package__ = 'WRF'
import unittest, re
import xarray as xr
import sys
from importlib.util import find_spec
from . import *

if find_spec('mpi4py'):
    MPI = import_module('mpi4py.MPI')
    WRF = import_module('.MPI', 'WRF')
    rank = MPI.COMM_WORLD.Get_rank()
else:
    WRF = import_module('.threads', 'WRF')
    rank = -1


# xarray.testing methods compare dimensions etc too, which we don't want here
class WRFTests(Application, unittest.TestCase):
    test_dir = Unicode().tag(config=True)

    interpolator = Unicode('scipy').tag(config=True)
    """which interpolator to use (see :mod:`.interpolate`)"""

    n_proc = Integer(3).tag(config=True)
    """Number of processors to use for the test"""

    init_dict = {}

    def __init__(self, *args, **kwargs):
        Application.__init__(self, **{k: kwargs.pop(k, self.init_dict.get(k, None))
                                      for k in ['parent', 'config']})
        unittest.TestCase.__init__(self, *args, **kwargs)
        if len(self.config) == 0:
            try:
                self.load_config_file(os.path.expanduser('~/Dropbox/work/config.py'))
            except: pass

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
        cc = WRF.Concatenator(domain='d03', interpolator=self.interpolator)
        if cc.rank == 0:
            cc.dirs = [d for d in cc.dirs if re.search('c01_2016120[1-3]', d)]
        cc.concat(**kwargs)
        comm.barrier()
        comm.Disconnect()

    def setUp(self):
        if rank < 1:
            self.test = xr.open_dataset(
                os.path.join(self.test_dir, 'WRF_201612_{}.nc'.format(self.__class__.__name__))
            )
        self.cc = WRF.Concatenator(domain='d03', outfile='out.nc', config=self.config)
        print(self.cc.config)
        self.cc.dirs = [d for d in self.cc.dirs if re.search('c01_2016120[1-3]', d)]

    def tearDown(self):
        self.data.close()
        self.test.close()

    def run_util(self, **kwargs):
        return None
        for k, v in kwargs.items():
            setattr(self.cc, k, v)
        if rank == -1:
            self.cc.concat()
            self.data = self.cc.data
        else:
            self.cc.start()
            self.cc.comm.barrier()
        if rank == 0:
            self.data = xr.open_dataset('out.nc')

    def start(self):
        self.init_dict = {'config': self.config}
        suite = unittest.defaultTestLoader.loadTestsFromName('WRF.tests')
        runner = unittest.TextTestRunner(sys.stderr)
        runner.run(suite)

class T2_all(WRFTests):
    def test_whole_domain(self):
        self.run_util(variables='T2')
        if rank == 0:
            test_data = self.test['field']
            np.testing.assert_allclose(self.data['T2'].transpose(*test_data.dims), test_data)

    def test_interpolated(self):
        self.run_util(variables='T2', interpolate=True)
        if rank == 0:
            test_data = self.test['interp']
            data = self.data['T2'].transpose(*test_data.dims).sel(station=test_data.station)
            np.testing.assert_allclose(data, test_data, rtol=1e-3)

class T2_lead1(WRFTests):
    def test_whole_domain(self):
        self.run_util(variables='T2', lead_day=1)
        if rank == 0:
            test_data = self.test['field']
            np.testing.assert_allclose(self.data['T2'].transpose(*test_data.dims), test_data)

    def test_interpolated(self):
        self.run_util(variables='T2', lead_day=1, interpolate=True)
        if rank == 0:
            test_data = self.test['interp']
            data = self.data['T2'].transpose(*test_data.dims).sel(station=test_data.station)
            np.testing.assert_allclose(data, test_data, rtol=1e-3)

class T_all(WRFTests):
    def test_whole_domain(self):
        self.run_util(variables='T')
        if rank == 0:
            test_data = self.test['field']
            np.testing.assert_allclose(self.data['T'].transpose(*test_data.dims), test_data)

    def test_interpolated(self):
        self.run_util(variables='T', interpolate=True)
        if rank == 0:
            test_data = self.test['interp']
            data = self.data['T'].transpose(*test_data.dims).sel(station=test_data.station)
            np.testing.assert_allclose(data, test_data, rtol=1e-3)

class T_lead1(WRFTests):
    def test_whole_domain(self):
        self.run_util(variables='T', lead_day=1)
        if rank == 0:
            test_data = self.test['field']
            np.testing.assert_allclose(self.data['T'].transpose(*test_data.dims), test_data)

    def test_interpolated(self):
        self.run_util(variables='T', lead_day=1, interpolate=True)
        if rank == 0:
            test_data = self.test['interp']
            data = self.data['T'].transpose(*test_data.dims).sel(station=test_data.station)
            np.testing.assert_allclose(data, test_data, rtol=1e-3)


# runner = unittest.TextTestRunner()

# n_proc = 2
# interpolator = 'bilinear'

# T2_suite = unittest.TestSuite()
# T2_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2_all))
# T2_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T2_lead1))

# T_suite = unittest.TestSuite()
# T_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T_all))
# T_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(T_lead1))
