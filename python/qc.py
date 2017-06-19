#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Shafer, Mark A., Christopher A. Fiebrich, Derek S. Arndt, Sherman E. Fredrickson, and Timothy W. Hughes. “Quality Assurance Procedures in the Oklahoma Mesonetwork.” Journal of Atmospheric and Oceanic Technology 17, no. 4 (2000): 474–494.


class QC(object):
    max_ts = 60
    def __init__(self, data, qparams):
        self.data = data
        self.q = qparams

    def _qual(self, x, name):
        y = xr.DataArray(x).expand_dims('check')
        y['check'] = [name]
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
        flags = flags.rename({'dim_0': 'time', 'dim_1': 'sensor_code'})
        flags.to_netcdf(path)

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
        self._qual(abs(self.data.xs('avg', 1, 'aggr') - ma + mi), 'range_avg')

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
        dxdt = pd.DataFrame(dx / dt, index = x.index[1:][i], columns = idx)
        return dxdt.reindex(y.index)

    def step(self, var, calibrate=False):
        "change between successive observations"
        q = self.q[(var, 'step')]
        dxdt = self.data.groupby(axis=1, level='sensor_code').apply(self._step, max_ts=self.max_ts)
        da, DA = self._align(dxdt.xs('avg', 1, 'aggr'), q['avg'])
        dx, DX = self._align(dxdt.xs('max', 1, 'aggr'), q['max'])
        if calibrate == True:
            self._qual(da, 'step_avg')
            self._qual(dx, 'step_max')
        else:
            self._qual(da > DA, 'step_avg')
            self._qual(dx > DX, 'step_max')

    def persistence(self, var, calibrate=False):
        q = self.q[(var, 'persist')]
        g = self.data.groupby(self.data.index.date)
        gmax = g.max()
        gmin = g.min()
        dmax = abs(gmax.xs('max', 1, 'aggr') - gmin.xs('min', 1, 'aggr'))
        davg = abs(gmax.xs('avg', 1, 'aggr') - gmin.xs('avg', 1, 'aggr'))

        # StD of min / max probably has different properties - for now use only ave
        st, ST = self._align(g.std().xs('avg', 1, 'aggr'), q['std'])
        dx, DX = self._align(dmax, q['max'])
        da, DA = self._align(davg, q['avg'])
        if calibrate == True:
            self._qual(st, 'persist_std')
            self._qual(dx / st, 'persist_max')
            self._qual(da / st, 'persist_avg')
        else:
            self._qual(st < ST, 'persist_std')
            self._qual(dx / st < DX, 'persist_max')
            self._qual(da / st < DA, 'persist_avg')

    def spatial(self):
        pass

    def _calplot(self, t, c, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        for r, x in self.data[c].iteritems():
            ax.plot(x.dropna(), label=r)
        ax.plot(self.data.loc[t, c], 'ro')
        ax.legend()
        # ax.set_title(d.columns.get_level_values('station').unique()[0])
        plt.pause(0.1)
        plt.show(block=False)
        return ax

    def where(self, df, aggr='avg', window = '1D', cont = False):
        if not cont:
            self.keep = pd.DataFrame()
            self.reject = pd.DataFrame()
            self.notes = pd.DataFrame()
            m = df.as_matrix().flatten()
            i = np.argsort(m)
            i, j = np.unravel_index(i[np.isfinite(m[i])][::-1], df.shape)
            self._where = zip(st.index[i], st.columns[j])
        while self._where:
            t, c = next(self._where)
            dt = pd.Timedelta(window)
            plt.figure(figsize=(10,5))
            d = self.data.xs(c, 1, 'sensor_code').xs(aggr, 1, 'aggr')
            plt.subplot(1, 2, 1)
            plt.title(d.columns.get_level_values('station')[0])
            plt.plot(d[t-dt:t+dt].dropna())
            plt.plot(t, float(d.loc[t]), 'ro')
            plt.subplot(1, 2, 2)
            plt.plot(d.dropna())
            plt.plot(t, float(d.loc[t]), 'ro')
            print(df.loc[t, c], np.diff(d.loc[:t].index[-2:]).astype('timedelta64[m]'))
            plt.pause(.1)
            inp = input('keep ([y]/n)?  ')
            rec = pd.DataFrame(df.loc[t, c], index = [t], columns = [c])
            if inp =='' or inp[0] != 'n':
                self.keep =rec.combine_first(self.keep)
            if len(inp) > 1:
                self.notes = pd.DataFrame(inp[1:], index = [t], columns = [c]).combine_first(self.notes)
            else:
                self.reject = rec.combine_first(self.reject)
            if input('continue ([y]/n)?  ') == 'n':
                self.last = rec
                break
            plt.close()

    def where_switch(self):
        try:
            r = self.keep.loc[self.last.index[0], self.last.columns]
            self.keep.loc[self.last.index[0], self.last.columns] = np.nan
            self.last.combine_first(self.reject)
            print('moved from keep to reject')
        except KeyError:
            r = self.reject.loc[self.last.index[0], self.last.columns]
            self.reject.loc[self.last.index[0], self.last.columns] = np.nan
            self.last.combine_first(self.keep)
            print('moved from reject to keep')

    def save_calibration(self, name, check):
        with pd.HDFStore(name) as s:
            for k in ['keep', 'reject', 'notes']:
                s['/{}/{}'.format(check, k)] = getattr(self, k)

    def plot_where(df, cond, start_from=None):
        cols = df[cond].dropna(1, 'all')
        for i, c in enumerate(cols.iteritems()):
            if start_from is not None and i + 1 < start_from:
                continue
            plt.figure()
            plt.plot(df[c[0]].dropna())
            plt.plot(c[1], 'ro')
            plt.title(c[0])
            plt.pause(.1)
            print(c[1].dropna())
            if input('{} of {}, continue ([y]/n)?  '.format(i + 1, cols.shape[1])) == 'n':
                break
            plt.close()

    def sensor_to_columns(self, s, aggr):
        return self.data.loc[:, pd.IndexSlice[:,:,s,:,aggr]].columns


    def global_calibr_range(self, var, limit):
        d = self.data.copy()

        I, J = [], []
        while True:
            m = d.max().max() if limit == 'max' else d.min().min()
            i, j = np.where(d == m)
            t = d.index[list(set(i))]
            c = d.columns[list(set(j))]
            s = c.get_level_values('sensor_code')[0]
            print('max value {:0.2f} at sensor {} and time {}'.format(m, s, t[0].isoformat()))

            inp = input('plot ([y]/n)?  ')
            if inp != 'n':
                ax = self._calplot(t, c)

            inp = input('remove and continue ([n]/y)?  ')
            if inp == 'y':
                d.loc[t, c] = np.nan
                I.append(i)
                J.append(j)
                self.q[(var, 'range', limit)] = m
            else:
                break
        return I, J



if __name__ == '__main__':
    import data
    D = data.Data()
    D.open('r','raw.h5')

    qp = D.sta['elev'].to_frame()
    i = qp[qp > '2000'].dropna().index
    qp.columns = [('elev', None, None)]
    qp[('ta_c', 'range', 'max')] = 40
    qp.loc[i, ('ta_c', 'range', 'max')] = 30
    qp[('ta_c', 'range', 'min')] = -25
    qp[('ta_c', 'step', 'avg')] = 0.1 # per minute
    qp[('ta_c', 'step', 'max')] = 0.1 # per minute
    qp[('ta_c', 'persist', 'std')] = 0.1
    qp[('ta_c', 'persist', 'max')] = 0.1
    qp[('ta_c', 'persist', 'avg')] = 0.1

    qp.columns = pd.MultiIndex.from_tuples(qp.columns)

    Q = QC(pd.concat([D.r[k] for k in D.r.keys()], 1), qp)
