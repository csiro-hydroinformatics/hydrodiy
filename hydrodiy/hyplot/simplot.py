
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hydata import dutils
from hyplot import putils


class Simplot(object):

    def __init__(self, \
        fig, \
        obs, \
        sim, \
        wateryear_start =7, \
        nfloods=4, \
        nbeforepeak = 30, \
        nafterpeak = 60):

        # Properties
        self.wateryear_start = wateryear_start

        # data
        self.idx = pd.notnull(obs) & (obs >= 0)
        
        if obs.name is None:
            obs = pd.Series(obs, name='obs')
        self.data = pd.DataFrame(obs)
        
        self.nsim = 1
        if sim.name is None:
            sim = pd.Series(sim, name='sim {0}'.format(self.nsim))

        self.data = self.data.join(sim)

        # Number of flood events
        self.nfloods = nfloods
        self.nbeforepeak = nbeforepeak
        self.nafterpeak = nafterpeak
        self._get_flood_indexes()

        # Figure to draw on
        self.fig = fig

        # Grid spec
        fig_ncols = 1 + nfloods/2
        fig_nrows = 3
        self.gs = gridspec.GridSpec(fig_nrows, fig_ncols,
                width_ratios=[1] * fig_ncols,
                height_ratios=[1] * (fig_nrows-1) + [0.5] * 1)


    def _get_flood_indexes(self):
        
        obs_tmp = self.data['obs'].copy()
        nval = len(obs_tmp)
        self.flood_idx = []

        for iflood in range(self.nfloods):
            imax = np.where(obs_tmp == obs_tmp.max())[0]
            i1 = max(0, imax - self.nbeforepeak)
            i2 = min(nval-1, imax + self.nafterpeak)
            idx = range(i1, i2+1)
            self.flood_idx.append(idx)

            obs_tmp.iloc[idx] = np.nan


    def add_sim(self, sim):

        self.nsim += 1
        if sim.name is None:
            sim = pd.Series(sim, name='sim {0}'.format(self.nsim))

        self.data = self.data.join(sim)


    def draw(self):

        # Draw flow duration curves
        ax = plt.subplot(self.gs[0, 0])
        self.draw_fdc(ax)

        ax = plt.subplot(self.gs[1, 0])
        self.draw_fdc(ax, xlog=True, ylog=False)

        # Draw flood events
        for iflood in range(self.nfloods):
            ix = iflood/2 % self.nfloods 
            iy = 1 + iflood % (self.nfloods/2)
            ax = plt.subplot(self.gs[ix, iy])
            self.draw_floods(ax, iflood)

        # Draw annual time series
        ax = plt.subplot(self.gs[2, 1:])
        self.draw_annual(ax)

        # Draw water balance
        ax = plt.subplot(self.gs[2, 0])
        self.draw_balance(ax)


    
    def draw_fdc(self, ax, xlog=False, ylog=True):
        
        if xlog:
            ax.set_xscale('log', nonposx='clip')

        if ylog:
            ax.set_yscale('log', nonposx='clip')

        # Freqency
        idx = self.idx
        nval = np.sum(idx)
        ff = (np.arange(1, nval+1)-0.3)/(nval+0.4)
        data = self.data

        for cn in data.columns:
            value = np.sort(data.loc[idx, cn].values)[::-1]
            ax.plot(ff, value, label=cn)
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Flow')
        ax.set_title('Flow duration curve')
        ax.legend(loc=1, frameon=False)
        ax.grid()


    def draw_floods(self, ax, iflood):

        data = self.data
        idx = self.flood_idx[iflood]
        dates = data.index.values[idx]

        data.loc[dates, :].plot(ax=ax)

        values = data.loc[dates, 'obs'].values
        yy = dates[values == np.max(values)][0]
        title = 'Flood #{0} - {1:%Y-%m}'.format(iflood+1, pd.to_datetime(yy))
        ax.set_title(title)
        ax.grid()

        ax.set_xlabel('Date')
        ax.set_ylabel('Flow')


    def draw_annual(self, ax):

        # Compute annual time series
        datam = self.data.apply(dutils.aggmonths, args=(1,))
        datay = datam.apply(pd.rolling_sum, args=(12,))

        im = self.wateryear_start - 1
        if im == 0:
            im = 12
        datay = datay.loc[datay.index.month == im,:]
        datay = datay.shift(-1)

        # plot
        datay.plot(ax=ax)

        ax.set_ylabel('Annual flow')
        ax.grid()


    def draw_balance(self, ax):
        data = self.data
        datab = data.loc[self.idx, :].mean()

        # plot
        datab.plot(ax=ax, kind='bar')

        ax.set_ylabel('Average flow')
        ax.set_title('Water Balance')
        ax.grid()



