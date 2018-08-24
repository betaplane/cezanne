#!/usr/bin/env python
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

    The multiple inheritance of :class:`WRFTests` might be a problem in the future. Currently it seems that unittest, when run with the idiom ``python -m unittest WRF.tests``, calls :meth:`WRFTests.__init__` with the test method name as argument. 

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
    """directory where the comparison test data resides"""

    interpolator = Unicode('scipy').tag(config=True)
    """which interpolator to use (see :mod:`.interpolate`)"""

    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    """path to file with (traitlets-type) config values for the tests"""

    test_name = Unicode('WRF.tests').tag(config=True)
    """dotted test name to be ingested by :meth:`unittest.TestLoader.loadTestsFromName`"""

    aliases = {'f': 'WRFTests.config_file', 'n': 'WRFTests.test_name', 'log_level': 'Application.log_level'}

    def __init__(self, *args, parent=None, config=None, **kwargs):
        Application.__init__(self, parent=parent)
        self.parse_command_line()
        try:
            self.load_config_file(os.path.expanduser(self.config_file))
        except ConfigFileNotFound:
            pass
        if config is not None:
            self.update_config(config)
        unittest.TestCase.__init__(self, *args, **kwargs)

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
        self.cc.dirs = [d for d in self.cc.dirs if re.search('c01_2016120[1-3]', d)]

    def run_util(self, test_var, rtol=None, **kwargs):
        for k, v in kwargs.items():
            setattr(self.cc, k, v)
        if rank == -1:                # for threads module
            self.cc.concat()
            self.data = self.cc.data
        else:                         # for MPI module
            self.cc.start()
            self.cc.comm.barrier()
        if rank < 1:
            data = xr.open_dataset('out.nc') if rank==0 else self.data
            test_data = self.test[test_var]
            if kwargs.get('interpolate', False):
                data = data.sel(station=test_data.station)
            if rtol is not None:
                np.testing.assert_allclose(data[kwargs.get('variables')].transpose(*test_data.dims), test_data, rtol=rtol)
            else:
                np.testing.assert_allclose(data[kwargs.get('variables')].transpose(*test_data.dims), test_data)
            self.log.info('Test %s performed', self)
            os.remove('out.nc')
            self.test.close()

    def _iter_config(self, i):
        try:
            return [self._iter_config(t) for t in i]
        except TypeError:
            i.update_config(self.config)

    def start(self):
        suite = unittest.defaultTestLoader.loadTestsFromName(self.test_name)
        self._iter_config(suite)
        runner = unittest.TextTestRunner()
        runner.run(suite)

class T2_all(WRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T2')

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T2', interpolate=True)

class T2_lead1(WRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T2', lead_day=1)

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T2', lead_day=1, interpolate=True)

class T_all(WRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T')

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T', interpolate=True)

class T_lead1(WRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T', lead_day=1)

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T', lead_day=1, interpolate=True)

if __name__ == '__main__':
    app = WRFTests()
    app.start()
