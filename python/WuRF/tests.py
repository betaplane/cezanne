#!/usr/bin/env python
"""
Tests
-----

Tests can be run in the :mod:`unittest` fashion **if** the config file contains all necessary data and the :mod:`.threadWuRF` submodule is targeted::

    python -m unittest WuRF.tests

or for an individual test case::

    python -m unittest WuRF.tests.<test_case>

For convenience, this module can also be run as a script, with the `traitlets  <https://traitlets.readthedocs.io/en/stable/config.html>`_-based command-line configuration system::

    ./tests.py -f path/to/config --log_level=INFO -n WuRF.tests.T2_all 2>&1

This form can also be run with mpirun/mpiexec (the tests should also pass even without invoking MPI, i.e. with one processor, **but don't use more than 3 processors**)::

    mpirun -n 3 ./tests.py -f path/to/config --log_level=INFO -n WuRF.tests.T2_all 2>&1

This demonstrates all available flags and redirection of the log output from stderr to stdout. To get help on the available configuration options, do ``./tests.py --help`` or ``--help-all``.

The tests can also be run with the help of the :class:`condor.SSHFSRunner` script runner, in which case :class:`.WuRFTests` needs to be added to the :attr:`condor.SSHFSRunner.subcommands` in importable (dotted) form ('WuRF.tests.WuRFTests'), e.g.::

    mpiexec -n 3 ./sshfs wrftest

.. NOTE::

    The multiple inheritance of :class:`WuRFTests` might be a problem in the future. Currently it seems that unittest, when run with the idiom ``python -m unittest WuRF.tests``, calls :meth:`WuRFTests.__init__` with the test method name as argument. 

"""
__package__ = 'WuRF'
import unittest, re
import xarray as xr
import sys
from importlib.util import find_spec
from .base import *

if find_spec('mpi4py'):
    MPI = import_module('mpi4py.MPI')
    WuRF = import_module('.mpiWuRF', 'WuRF')
    rank = MPI.COMM_WORLD.Get_rank()
else:
    WuRF = import_module('.threadWuRF', 'WuRF')
    rank = -1


# xarray.testing methods compare dimensions etc too, which we don't want here
class WuRFTests(Application, unittest.TestCase):
    test_dir = Unicode().tag(config=True)
    """directory where the comparison test data resides"""

    interpolator = Unicode('scipy').tag(config=True)
    """which interpolator to use (see :mod:`.interpolate`)"""

    config_file = Unicode('~/Dropbox/work/config.py').tag(config=True)
    """path to file with (traitlets-type) config values for the tests"""

    test_name = Unicode('WuRF.tests').tag(config=True)
    """dotted test name to be ingested by :meth:`unittest.TestLoader.loadTestsFromName`"""

    aliases = {'f': 'WuRFTests.config_file', 'n': 'WuRFTests.test_name', 'log_level': 'Application.log_level'}

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

    def setUp(self):
        if rank < 1:
            self.test = xr.open_dataset(
                os.path.join(self.test_dir, 'WRF_201612_{}.nc'.format(self.__class__.__name__))
            )
        self.cc = WuRF.CC(domain='d03', outfile='out.nc', config=self.config)
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

class T2_all(WuRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T2')

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T2', interpolate=True)

class T2_lead1(WuRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T2', lead_day=1)

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T2', lead_day=1, interpolate=True)

class T_all(WuRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T')

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T', interpolate=True)

class T_lead1(WuRFTests):
    def test_whole_domain(self):
        self.run_util('field', variables='T', lead_day=1)

    def test_interpolated(self):
        self.run_util('interp', 1e-3, variables='T', lead_day=1, interpolate=True)

if __name__ == '__main__':
    app = WuRFTests()
    app.start()
