import unittest, re
import xarray as xr
import numpy as np
import sys, os
from mpi4py import MPI
from importlib import import_module
from . import config, WRF


# I can't get the xarray.testing method of the same name to work (fails due to timestamps)
class WRFTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.remove('out.nc')

    @staticmethod
    def run_concat():
        comm = MPI.Comm.Get_parent()
        kwargs = None
        kwargs = comm.bcast(kwargs, root=0)
        cc = WRF.Concatenator('d03', interpolator=None)
        if cc.rank == 0:
            cc.files.dirs = [d for d in cc.files.dirs if re.search('c01_2016120[1-3]', d)]
        cc.concat(**kwargs)
        comm.barrier()
        comm.Disconnect()

class TestField(WRFTests):
    def test_all_whole_domain(self):
        comm = MPI.COMM_SELF.Spawn(sys.executable, ['-c', 'from data import tests;tests.WRFTests.run_concat()'], maxprocs=3)
        comm.bcast({'variables': 'T2'}, root=MPI.ROOT)
        comm.barrier()
        comm.Disconnect()
        with xr.open_dataset(config['tests']['all_days']) as data:
            with xr.open_dataset('out.nc') as out:
                np.testing.assert_allclose(
                    out['T2'].transpose('start', 'Time', 'south_north', 'west_east'),
                    data['field'].transpose('start', 'Time', 'south_north', 'west_east'))

    def test_lead_day_whole_domain(self):
        comm = MPI.COMM_SELF.Spawn(sys.executable, ['-c', 'from data import tests;tests.WRFTests.run_concat()'], maxprocs=3)
        comm.bcast({'variables': 'T2', 'lead_day': 1}, root=MPI.ROOT)
        comm.barrier()
        comm.Disconnect()
        with xr.open_dataset(config['tests']['lead_day1']) as data:
            with xr.open_dataset('out.nc') as out:
                np.testing.assert_allclose(
                    out['T2'].transpose('Time', 'south_north', 'west_east'),
                    data['field'].transpose('time', 'south_north', 'west_east'))

class TestIntp(WRFTests):
    def test_all_interpolated(self):
        with xr.open_dataset(config['tests']['all_days']) as data:
            self.wrf.concat('T2', True)
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('start', 'station', 'Time'),
                data['interp'].transpose('start', 'station', 'Time'), rtol=1e-3)


    def test_lead_day_interpolated(self):
        with xr.open_dataset(config['tests']['lead_day1']) as data:
            self.wrf.concat('T2', True, 1)
            np.testing.assert_allclose(
                self.wrf.data['T2'].transpose('station', 'time'),
                data['interp'].transpose('station', 'time'), rtol=1e-4)

def run_tests():
    suite = unittest.TestSuite()
    # suite.addTests([Tests(t) for t in Tests.__dict__.keys() if t[:4]=='test'])
    suite.addTest(TestField('test_all_whole_domain'))
    suite.addTest(TestField('test_lead_day_whole_domain'))
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestField))
    runner = unittest.TextTestRunner()
    runner.run(suite)

