from ftplib import FTP
from glob import glob
import os, re
import numpy as np
import xarray as xr


class Downloader(object):
    def __init__(self, ftp_path='pub/cdp/159/688', local_path='.', files=None):
        self.login(ftp_path)
        if files is None:
            self.list()
        else:
            self.files = files

        for f in os.listdir(local_path):
            if f[:6] == 'patmos':
                self.files.remove(f)
        self.local_path = local_path

    def login(self, path):
        self.ftp = FTP('ftp.ssec.wisc.edu', 'anonymous', 'user@internet')
        self.ftp.login()
        self.ftp.cwd(path)

    def list(self):
        self.files = []
        self.ftp.retrlines('NLST', lambda l: self.files.append(l))

    def get(self):
        for i, f in enumerate(self.files):
            print('fetching {}'.format(f))
            self.ftp.retrbinary('RETR {}'.format(f), open(os.path.join(self.local_path, f), 'wb').write)
            if (i + 1) % 100 == 0:
                self.concat_files()

    def __del__(self):
        self.ftp.close()

    def concat_files(self):
        patterns = np.unique([f.split('_')[:4] for f in os.listdir(self.local_path)], axis=0)
        for p in patterns:
            s = '_'.join(p)
            l = glob(os.path.join(self.local_path, s + '*nc'))
            ds = xr.open_mfdataset(l)
            ds.to_netcdf('{}_{}.nc'.format(s, ds.indexes['time'].max().strftime('%Y%m%d')))
            for f in l:
                os.remove(f)
