import warnings

import re
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from calendar import month_abbr as months

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
COLORS = putils.COLORS10


class Simplot(object):

    def __init__(self, \
        obs, \
        sim, \
        sim_name = None,
        fig = None, \
        nfloods = 3, \
        wateryear_start =7, \
        ndays_beforepeak = 30, \
        ndays_afterpeak = 60):

        # Properties
        self.wateryear_start = wateryear_start

        # data
        self.idx_obs = pd.notnull(obs) & (obs >= 0)
        obs.name = 'obs'
        self.data = pd.DataFrame(obs)
        self.nsim = 0
        self.sim_names=[]
        self.add_sim(sim, name=sim_name)

        self._compute_idx_all()

        # Number of flood events
        self.nfloods = nfloods
        self.ndays_beforepeak = ndays_beforepeak
        self.ndays_afterpeak = ndays_afterpeak
        self._get_flood_indexes()

        # Figure to draw on
        if fig is None:
            fig = plt.figure()
        self.fig = fig

        # Grid spec
        fig_ncols = 3
        fig_nrows = 3 + (nfloods-1)/3
        self.gs = gridspec.GridSpec(fig_nrows, fig_ncols,
                width_ratios=[1] * fig_ncols,
                height_ratios=[0.5] * 1 + [1] * (fig_nrows-1))


    def _compute_idx_all(self):
        ''' Compute indexes where all data are available'''
        self.idx_all = pd.isnull(self.data).sum(axis=1) == 0


    def _get_flood_indexes(self):
        ''' Identify flood events '''
        dates = self.data.index
        obs_tmp = self.data['obs'].copy()
        nval = len(obs_tmp)
        self.flood_idx = []
        iflood = 0

        for iflood in range(self.nfloods):
            idx = pd.notnull(obs_tmp)
            if np.sum(idx) == 0:
                continue

            date_max = obs_tmp[idx].argmax()
            idx = dates >= date_max - delta(days=self.ndays_beforepeak)
            idx = idx & (dates <= date_max + delta(days=self.ndays_afterpeak))

            if np.any(self.idx_all[idx]):
                self.flood_idx.append({'index':idx, 'date_max':date_max})

            obs_tmp.loc[idx] = np.nan

        if len(self.flood_idx) == 0:
            raise ValueError('Could not identify valid flood data')


    def _getname(self, cn):
        ''' Return a proper name for variable '''
        return re.sub('_', '\n', cn)


    def add_sim(self, sim, name=None):
        ''' Add a new simulations '''

        self.nsim += 1

        # Set sim name
        if name is None:
            name = sim.name

        if name is None or name == '':
            name = 'sim_{0:02}'.format(self.nsim)
        self.sim_names.append(name)

        # Store data
        self.data[name] = sim

        self._compute_idx_all()


    def draw(self):
        ''' Draw all plots '''
        # Draw water balance
        ax = plt.subplot(self.gs[0, 0])
        self.draw_balance(ax)

        # Draw annual time series
        ax = plt.subplot(self.gs[0, 1:])
        self.draw_annual(ax)

        # Draw flow duration curves
        ax = plt.subplot(self.gs[1, 0])
        self.draw_fdc(ax)

        ax = plt.subplot(self.gs[1, 1])
        self.draw_fdc(ax, 'd', xlog=True, ylog=False)

        # Draw seasonal residuals
        ax = plt.subplot(self.gs[1, 2])
        self.draw_monthlyres(ax, 'e')


        # Draw flood events
        for iflood in range(self.nfloods):
            ix = 2 + iflood/3
            iy = iflood%3
            ax = plt.subplot(self.gs[ix, iy])
            self.draw_floods(ax, iflood, letters[5+iflood])


    def draw_monthlyres(self, ax, ax_letter='d'):
        ''' Draw the plot of monthly residuals '''

        datam = self.data.loc[self.idx_all, :].resample('MS', how='sum')
        datam = datam.groupby(datam.index.month).mean()
        datam.columns = [self._getname(cn) for cn in  datam.columns]

        datam.plot(ax=ax, color=COLORS, marker='o', lw=3)

        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=2, frameon=False)

        ax.set_xlim((0, 13))
        ax.set_xticks(range(1, 13))
        ax.grid()
        ax.set_xlabel('Month')
        ax.set_ylabel('Flow')
        title = '({0}) Mean monthly simulations'.format(ax_letter)
        ax.set_title(title)


    def draw_fdc(self, ax, ax_letter='c', xlog=False, ylog=True):
        ''' Draw the flow duration curve '''

        if xlog:
            ax.set_xscale('log', nonposx='clip')

        if ylog:
            ax.set_yscale('log', nonposx='clip')

        # Freqency
        data = self.data
        idx = self.idx_all
        nval = np.sum(idx)
        ff = sutils.ppos(nval)

        icol = 0
        for cn in data.columns:
            value = np.sort(data.loc[idx, cn].values)[::-1]
            name = self._getname(cn)
            ax.plot(ff, value, '+-', label=name,
                markersize=6,
                color=COLORS[icol], lw=1)
            icol += 1

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Flow')

        title = '({0}) Flow duration curve'.format(ax_letter)
        if ylog:
            title += ' - Low flow'
        if xlog:
            title += ' - High flow'
        ax.set_title(title)
        ax.legend(loc=1, frameon=False)
        ax.grid()


    def draw_floods(self, ax, iflood, ax_letter):

        data = self.data
        idx = self.flood_idx[iflood]['index']

        dataf = data.loc[idx, :]
        dataf.columns = [self._getname(cn) for cn in  data.columns]

        dataf.plot(ax=ax, color=COLORS, lw=2, \
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
        ym = months[(self.wateryear_start-2)%12+1].upper()
        datay = self.data.resample('A-'+ym, how='sum')

        # plot - exclude first and last year to avoid missing values
        datay.iloc[1:-1, :].plot(ax=ax, color=COLORS, marker='o', lw=3)

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
        cc = [self._getname(cn) for cn in  data.columns]
        datab = pd.Series(data.loc[self.idx_all, :].mean().values,
                    index=cc)

        freq = data.index.freqstr
        freqfact = {'H':24*365.25, 'D':365.25, 'MS':12, 'ME':12}
        if freq not in freqfact:
            warnings.warn('Frequency [{0}] not recognised, assuming daily'.format(freq))
            fact = freqfact['D']
        else:
            fact = freqfact[freq]

        datab = datab * fact

        # plot
        datab.plot(ax=ax, kind='bar', color=COLORS, edgecolor='none')

        ax.set_ylabel('Mean annual')
        ax.set_title('({0}) Water Balance'.format(ax_letter))
        ax.grid()


    def savefig(self, filename, size=None):
        if size is None:
            size = (18, 10+5*((self.nfloods-1)/3+1))

        self.fig.set_size_inches(size)
        self.gs.tight_layout(self.fig)
        self.fig.savefig(filename)

