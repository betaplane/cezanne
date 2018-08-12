from mpi4py.futures import MPIPoolExecutor
from glob import glob

if __name__ == '__main__':
    from .WRF import Files

    F = Files(paths, hour, from_date)
    self.files = [f for d in F.dirs for f in glob(join(d, glob_pattern))]
    assert len(self.files) > 0, "no directories added"

    with MPIPoolExecutor() as exe:
        
