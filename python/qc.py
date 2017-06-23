#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Shafer, Mark A., Christopher A. Fiebrich, Derek S. Arndt, Sherman E. Fredrickson, and Timothy W. Hughes. “Quality Assurance Procedures in the Oklahoma Mesonetwork.” Journal of Atmospheric and Oceanic Technology 17, no. 4 (2000): 474–494.


class QC(object):
    max_ts = 30
    min_count = 24

    def __init__(self, data, qparams):
        self.data = data
        self.q = qparams

    def _qual(self, x, name):
        y = xr.DataArray(x).expand_dims('check').rename({'dim_0': 'time'})
        y['check'] = [name]
        y['time'] = pd.DatetimeIndex(y['time'].values)
        # y = y.astype(float).fillna(0)
        if hasattr(self, 'flags'):
            self.flags = xr.concat((self.flags, y), 'check')
        else:
            self.flags = y
        print('run check {}'.format(name))

    def q_by_station(self, check, operation, sta=None):
        """
        Return minimum, maximum or average of continuous quality measure (saved in QC.flags, obtained by running check with calibrate == True) by station.

        :param check: what measure in QC.flags to select
        :param operation: min, max or avg
        :param sta: data.sta station meta data

        """
        q = {'min': lambda x:x.min('dim_0'),
             'max': lambda x:x.max('dim_0'),
             'avg': lambda x:x.mean('dim_0')}[operation](
                 self.flags.sel(check=check)
                 )
        q['dim_1'] = self.flags['dim_1']['station']
        q = q.to_dataframe(check).drop('check', 1)
        return q if sta is None else q.join(sta)

    def to_dataframe(self, check):
        fl = self.flags.sel(check=check).copy()
        fl['dim_1'] = fl['dim_1']['sensor_code']
        fl = fl.to_dataframe(check).drop('check', 1).unstack()
        fl.columns = self.flags['dim_1'].to_index()
        return fl

    def to_netcdf(self, path):
        flags = self.flags.copy()
        flags['dim_1'] = self.flags['dim_1']['sensor_code']
        flags = flags.rename({'dim_1': 'sensor_code'})
        flags.to_dataset(name='flags').to_netcdf(path)

    @staticmethod
    def ds_to_dataframe(ds, check):
        df = ds.sel(check=check).to_dataframe().drop('check', 1)\
                                               .swaplevel(i='time', j='sensor_code').unstack()
        df.columns = df.columns.get_level_values('sensor_code')
        return df


    @staticmethod
    def _align(x, qp):
        return x.align(qp, axis=1, level='station', broadcast_axis=0)

    def range(self, var, calibrate=False):
        "values that fall outside climatological range"
        q = self.q[(var, 'range')]
        mi, MI = self._align(self.data.xs('min', 1, 'aggr'), q['min'])
        ma, MA = self._align(self.data.xs('max', 1, 'aggr'), q['max'])
        if calibrate == True:
            self._qual(mi, 'range_min')
            self._qual(ma, 'range_max')
        else:
            self._qual(mi < MI, 'range_min')
            self._qual(ma > MA, 'range_max')

        # also compute difference between mean and (max - min)
        # self._qual(abs(self.data.xs('avg', 1, 'aggr') - ma + mi), 'range_avg')

    # to be applied on a per-sensor basis, due to different time sampling
    # (presumably per-station, but per-sensor makes it simpler to code)
    @staticmethod
    def _step(y, max_ts):
        x = y.dropna()
        cols = list(y.columns.tolist()[0][:-1])
        print(cols)
        dt = np.diff(np.array(x.index, dtype='datetime64[m]').astype(int))

        # exclude steps longer than a threshold (i.e., missing values)
        i = (dt <= max_ts)
        dt = dt[i].reshape((-1, 1))
        x1 = x.iloc[:-1][i]
        x2 = x.iloc[1:][i]
        m = lambda z: z.as_matrix()

        # compute difference for average values
        dxavg = abs(m(x2.xs('avg', 1, 'aggr')) - m(x1.xs('avg', 1, 'aggr')))

        # but also for max / min (largest pairwise absolute difference)
        dxmax = np.r_['1', abs(m(x1.xs('max', 1, 'aggr')) - m(x2.xs('min', 1, 'aggr'))),
                        abs(m(x2.xs('max', 1, 'aggr')) - m(x1.xs('min', 1, 'aggr')))].max(1)

        # include dummy 'min' column so as to allow 'apply' operation (needs 3 columns to be returned)
        dx = np.r_['1', dxavg, dxmax.reshape((-1, 1)), np.zeros_like(dxavg)]
        idx = pd.MultiIndex.from_arrays(np.r_[cols, ['avg'], cols, ['max'], cols, ['min']].reshape((3,-1)).T)
        # dxdt = pd.DataFrame(dx / dt, index = x.index[1:][i], columns = idx)
        dxdt = pd.DataFrame(dx, index = x.index[1:][i], columns = idx)
        return dxdt.reindex(y.index)

    def step(self, var, calibrate=False):
        "change between successive observations"
        q = self.q[(var, 'step')]
        dxdt = self.data.groupby(axis=1, level='sensor_code').apply(self._step, max_ts=self.max_ts)
        da, DA = self._align(dxdt.xs('avg', 1, 'aggr'), q['avg'])
        # dx, DX = self._align(dxdt.xs('max', 1, 'aggr'), q['max'])
        if calibrate == True:
            self._qual(da, 'step_avg')
            self._qual(dx, 'step_max')
        else:
            self._qual(da > DA, 'step_avg')
            # self._qual(dx > DX, 'step_max')

    # persistence seems mostly affected by missing values - i.e. few values per interval - low std
    def persistence(self, var, calibrate=False):
        q = self.q[(var, 'persist')]
        g = self.data.groupby(self.data.index.date)
        gmax = g.max()
        gmin = g.min()
        c = g.count().xs('avg', 1, 'aggr')
        dmax = abs(gmax.xs('max', 1, 'aggr') - gmin.xs('min', 1, 'aggr'))
        davg = abs(gmax.xs('avg', 1, 'aggr') - gmin.xs('avg', 1, 'aggr'))

        # StD of min / max probably has different properties - for now use only ave
        st, ST = self._align(g.std().xs('avg', 1, 'aggr'), q['std'])
        # dx, DX = self._align(dmax, q['max'])
        da, DA = self._align(davg, q['avg'])
        # ct, ST = self._align(g.count().xs('avg', 1, 'aggr'), q['std'])
        st1 = st[(st > ST) & (c > self.min_count)]
        if calibrate == True:
            self._qual(st[(c > self.min_count) & (st > 0)], 'persist_std')
            # self._qual(ct, 'persist_count')
            self._qual(dx / st1, 'persist_max')
            self._qual(da / st1, 'persist_avg')
        else:
            self._qual(((st < ST) & (c > self.min_count)) | (st == 0), 'persist_std')
            # self._qual(dx / st1 < DX, 'persist_max')
            self._qual(da / st1 < DA, 'persist_avg')

    def spatial(self):
        pass

    def calibration(self, df, aggr = 'avg', sort = 'desc', window = '1D', cont = False, init=False, aux=None):
        """Use data produced from calls to test methods with param **calibration = True** to collect some limiting values. The routine steps through the sorted data point by point and asks whether to keep or reject the point (keep here means to keep as potential outlier, reject would indicate a regular point). The results are appended to the arrays QC,keep and QC.reject (they can be written to file by QC.save_calibration()). For each point a plot is shown with the immediate neighborhood (whose width is controlled by param **window**) and the whole time series. 

        :param df: data from calls to test methods with param **calibration = True**
        :type df: pandas.DataFrame as returned by call to QC.ds_to_dataframe()
        :param aggr: what aggregation level to use for the plots (has no other effect)
        :param sort: whether to step through the data in descending (**desc**, default) or ascending (**asc**) direction
        :type sort: str 'asc' or 'desc'
        :param window: window half-width for the detail plot default one day
        :param cont: whether to continue with a previously stopped step-through
        :param init: whether to start a step-through from scratch (i.e. initialize the QC.keep and QC.reject arrays)

        """
        if init or not hasattr(self, 'keep'):
            self.keep = pd.DataFrame()
            self.reject = pd.DataFrame()
        if not cont:
            m = df.as_matrix().flatten()
            i = np.argsort(m)
            if sort == 'desc':
                i = i[::-1]
            i, j = np.unravel_index(i[np.isfinite(m[i])], df.shape)
            self._where = zip(df.index[i], df.columns[j])
        while self._where:
            t, c = next(self._where)
            self.plot_series(t, c, aggr, window)
            d = self.data.xs(c, 1, 'sensor_code').xs(aggr, 1, 'aggr')
            if aux is None:
                print(df.loc[t, c], np.diff(d.loc[:t].dropna().index[-2:]).astype('timedelta64[m]'))
            else:
                print(df.loc[t, c], aux.loc[t, c])
            rec = pd.DataFrame(df.loc[t, c], index = [t], columns = [c])
            if input('keep ([y]/n)?  ') == 'n':
                self.reject = rec.combine_first(self.reject)
            else:
                self.keep = rec.combine_first(self.keep)
            if input('continue ([y]/n)?  ') == 'n':
                self.last = rec
                break
            plt.close()

    def plot_series(self, t, c, aggr = 'avg', window = '1D'):
        dt = pd.Timedelta(window)
        plt.figure(figsize=(10,5))
        d = self.data.xs(c, 1, 'sensor_code').xs(aggr, 1, 'aggr')
        plt.subplot(1, 2, 1)
        plt.title(d.columns.get_level_values('station')[0])
        plt.plot(d[t-dt:t+dt].dropna(), '-x')
        plt.plot(t, float(d.loc[t]), 'ro')
        plt.gca().set_xlim(t-dt, t+dt)
        plt.subplot(1, 2, 2)
        plt.plot(d.dropna())
        plt.plot(t, float(d.loc[t]), 'ro')
        plt.pause(.1)

    def switch_calibration(self, t=None, c=None):
        if t is None:
            t, c = self.last.index[0], self.last.columns[0]
        try:
            r = self.keep.loc[t, c]
            if np.isnan(r):
                raise Exception()
            self.keep.loc[t, c] = np.nan
            self.reject.loc[t, c] = r
            print('moved from keep to reject')
        except KeyError:
            r = self.reject.loc[t, c]
            if np.isnan(r):
                raise Exception()
            self.reject.loc[t, c] = np.nan
            self.keep.loc[t, c] = r
            print('moved from reject to keep')

    def save_calibration(self, name, check):
        with pd.HDFStore(name) as s:
            for k in ['keep', 'reject']:
                s['/{}/{}'.format(check, k)] = getattr(self, k)

    def load_calibration(self, name, check):
        with pd.HDFStore(name) as s:
            for k in ['keep', 'reject']:
                setattr(self, k, s['/{}/{}'.format(check, k)])

    def plot_calibration(self, sta, flds, **kwargs):
        idx = pd.IndexSlice
        plt.figure()
        for k in ['keep', 'reject']:
            x = kwargs[k] if k in kwargs else getattr(self, k).copy()
            x.columns = flds.loc[idx[:, :, x.columns], :].index.droplevel('field')
            x = x.T.stack().to_frame().join(sta['elev'])
            plt.scatter(x[0], x['elev'])


if __name__ == '__main__':
    import data
    D = data.Data()
    D.open('r','raw.h5')

    qp = D.sta['elev'].astype(float).to_frame()
    qp.columns = [('elev', None, None)]
    qp[('ta_c', 'range', 'max')] = 38.5
    qp[('ta_c', 'range', 'min')] = np.minimum(-5, qp.iloc[:,0] * (-0.007367) + 5)
    qp[('ta_c', 'step', 'avg')] = 10 # absolute
    qp[('ta_c', 'persist', 'std')] = 0.1
    qp[('ta_c', 'persist', 'avg')] = 2

    qp.columns = pd.MultiIndex.from_tuples(qp.columns)

    Q = QC(pd.concat([D.r[k] for k in D.r.keys()], 1), qp)
    Q.range('ta_c')
    Q.step('ta_c')
    Q.persistence('ta_c')
    Q.to_netcdf('../../data/CEAZAMet/quality_ta_c.nc')
