import numpy as np
import pandas as pd
from statsmodels.tsa import ar_model

class regress(object):
    def __init__(self, z):
        self.r = ar_model.AR(z['smap'], z.index).fit(1).resid.to_frame()
        self.i = self.r[(self.r > self.r.std()) & (self.r.shift(-1) > 0)].dropna().index
        x = self.r.loc[self.i]
        # x = pd.concat((self.r, self.r.shift(-1)), 1).loc[self.i]
        # x = pd.concat((self.r, z['temp']), 1).loc[self.i]
        self.b = np.linalg.lstsq(x, z.loc[self.i, 'ceaza'])
        self.x = pd.concat((z['ceaza'], x.dot(self.b[0])), 1)

def plot(ij, stations, r, ax, color='w'):
    plt.plot(r.x.index, r.x['ceaza'], color='lightgreen')
    plt.plot(r.x.index, r.x[0].fillna(0), color='magenta')
    ax.set_xticks([])
    ylim = ax.get_ylim()
    dy = np.diff(ylim).item()
    plt.text(ax.get_xlim()[0]+.1, ylim[1]-dy/4, ', '.join(stations), color=color)
    cplots.axesColor(ax, color)



if __name__ == '__main__':
    zz = []
    for p, s in ij.items():
        try:
            z = cell(p, s)
            zz.append(regress(z))
            print(s)
        except:
            print(s, ' failed')
            zz.append(False)

    fig = plt.figure(figsize=(10, 9))
    fig.subplots_adjust(wspace=.3)
    for k, (p, s) in enumerate(ij.items()):
        if not zz[k]: continue
        ax = plt.subplot(11, 2, k+1)
        plot(p, s, zz[k], ax, 'w' if coastal[k] else 'orange')
        ax.set_xlim('2015-08', '2017-07-10')
