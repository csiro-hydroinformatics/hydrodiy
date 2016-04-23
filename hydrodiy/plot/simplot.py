import warnings

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from string import ascii_letters as letters

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors

from hydrodiy.data import dutils
from hydrodiy.stat import sutils
from hydrodiy.plot import putils

# Select color scheme
sim_colors = putils.tableau_colors


class Simplot(object):

    def __init__(self, \
        obs, \
        sim, \
        fig = None, \
        wateryear_start =7, \
        nfloods=4, \
        nbeforepeak = 30, \
        nafterpeak = 60):

        # Properties
        self.wateryear_start = wateryear_start

        # data
        self.idx_obs = pd.notnull(obs) & (obs >= 0)
        obs = pd.Series(obs.values, name='obs', index=obs.index)
        self.data = pd.DataFrame(obs)

        self.nsim = 1
        sim = pd.Series(sim.values, name='sim_{0:02d}'.format(self.nsim),
                    index=sim.index)
        self.data = self.data.join(sim)

        self._compute_idx_all()

        # Number of flood events
        self.nfloods = nfloods
        self.nbeforepeak = nbeforepeak
        self.nafterpeak = nafterpeak
        self._get_flood_indexes()

        # Figure to draw on
        if fig is None:
            fig = plt.figure()
        self.fig = fig

        # Grid spec
        fig_ncols = 1 + nfloods/2
        fig_nrows = 3
        self.gs = gridspec.GridSpec(fig_nrows, fig_ncols,
                width_ratios=[1] * fig_ncols,
                height_ratios=[0.5] * 1 + [1] * (fig_nrows-1))

    def _compute_idx_all(self):

        self.idx_all = self.data.apply(lambda x:
                                        np.all(pd.notnull(x)), axis=1)


    def _get_flood_indexes(self):

        obs_tmp = self.data['obs'].copy()
        sim = self.data['sim_01']
        dates = self.data.index.astype(datetime).values
        nval = len(obs_tmp)
        self.flood_idx = []
        iflood = 0

        while iflood < self.nfloods:
            imax = np.where(obs_tmp == obs_tmp.max())[0]
            date_max = pd.to_datetime(dates[imax][0])
            idx = dates >= date_max - delta(days=self.nbeforepeak)
            idx = idx & (dates <= date_max + delta(days=self.nafterpeak))

            if np.any(pd.notnull(sim[idx])):
                self.flood_idx.append({'index':idx, 'date_max':date_max})
                iflood += 1

            obs_tmp.loc[idx] = np.nan


    def add_sim(self, sim):

        self.nsim += 1
        if sim.name is None:
            sim = pd.Series(sim.values, name='sim_{0:02d}'.format(self.nsim),
                        index=sim.index)

        self.data = self.data.join(sim)

        self._compute_idx_all()


    def draw(self):

        # Draw water balance
        ax = plt.subplot(self.gs[0, 0])
        self.draw_balance(ax)

        # Draw annual time series
        ax = plt.subplot(self.gs[0, 1:])
        self.draw_annual(ax)

        # Draw flow duration curves
        ax = plt.subplot(self.gs[1, 0])
        self.draw_fdc(ax)

        ax = plt.subplot(self.gs[2, 0])
        self.draw_fdc(ax, 'd', xlog=True, ylog=False)

        # Draw flood events
        for iflood in range(self.nfloods):
            ix = 1 + iflood/2 % self.nfloods
            iy = 1 + iflood % (self.nfloods/2)
            ax = plt.subplot(self.gs[ix, iy])
            self.draw_floods(ax, iflood, letters[4+iflood])



    def draw_fdc(self, ax, ax_letter='c', xlog=False, ylog=True):

        if xlog:
            ax.set_xscale('log', nonposx='clip')

        if ylog:
            ax.set_yscale('log', nonposx='clip')

        # Freqency
        data = self.data
        idx = self.idx_all
        nval = np.sum(idx)
        ff = sutils.empfreq(nval)

        icol = 0
        for cn in data.columns:
            value = np.sort(data.loc[idx, cn].values)[::-1]
            ax.plot(ff, value, '-', label=cn, color=sim_colors[icol], lw=3)
            icol += 1

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Flow')

        title = '({0}) Flow duration curve'.format(ax_letter)
        if ylog:
            title += '- Low flow'
        if xlog:
            title += '- High flow'
        ax.set_title(title)
        ax.legend(loc=1, frameon=False)
        ax.grid()


    def draw_floods(self, ax, iflood, ax_letter):

        data = self.data
        idx = self.flood_idx[iflood]['index']
        data.loc[idx, :].plot(ax=ax, color=sim_colors, lw=2, \
                marker='o', legend=iflood==0)

        if iflood == 0:
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels, loc=2, frameon=False)

        date_max = self.flood_idx[iflood]['date_max']
        title = '({0}) Flood #{1} - {2:%Y-%m}'.format(ax_letter, \
            iflood+1, date_max)
        ax.set_title(title)
        ax.grid()
        ax.set_ylabel('Flow')


    def draw_annual(self, ax, ax_letter='b'):

        # Compute annual time series
        datam = self.data.apply(dutils.aggmonths, args=(1,))
        datay = datam.apply(pd.rolling_sum, args=(12,))

        im = self.wateryear_start - 1
        if im == 0:
            im = 12
        datay = datay.loc[datay.index.month == im,:]
        datay = datay.shift(-1)

        # plot
        datay.iloc[:-1, :].plot(ax=ax, color=sim_colors, marker='o', lw=3)

        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=2, frameon=False)

        month = datetime(1900, self.wateryear_start, 1).strftime('%B')
        title = '({0}) Annual time series - Start of water year in {1}'.format( \
                    ax_letter, month)
        ax.set_title(title)
        ax.set_ylabel('({0}) Annual flow'.format(ax_letter))
        ax.grid()


    def draw_balance(self, ax, ax_letter='a'):
        data = self.data
        datab = data.loc[self.idx_all, :].mean()

        freq = data.index.freqstr
        freqfact = {'H':24*365.25, 'D':365.25, 'MS':12, 'ME':12}
        if freq not in freqfact:
            warnings.warn('Frequency [{0}] not recognised, assuming daily'.format(freq))
            fact = freqfact['D']
        else:
            fact = freqfact[freq]
        datab = datab * fact

        # plot
        datab.plot(ax=ax, kind='bar', color=sim_colors, edgecolor='none')

        ax.set_ylabel('Average flow')
        ax.set_title('({0}) Water Balance'.format(ax_letter))
        ax.grid()


    def savefig(self, filename, size=(15, 15)):
        self.fig.set_size_inches(size)
        self.gs.tight_layout(self.fig)
        self.fig.savefig(filename)

