import warnings

import re
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
        sim_name = None,
        fig = None, \
        nfloods = 3, \
        wateryear_start =7, \
        nbeforepeak = 30, \
        nafterpeak = 60):

        # Properties
        self.wateryear_start = wateryear_start

        # data
        self.idx_obs = pd.notnull(obs) & (obs >= 0)
        obs = pd.Series(obs.values, name='obs', index=obs.index)
        self.data = pd.DataFrame(obs)
        self.nsim = 0
        self.sim_names=[]
        self.add_sim(sim, sim_name=sim_name)

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
        fig_ncols = 3
        fig_nrows = 2 + nfloods/3
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

    def _getname(self, cn):
        if cn.startswith('sim'):
            k = int(re.sub('sim_', '', cn))-1
            return self.sim_names[k]
        else:
            return cn

    def add_sim(self, sim, sim_name=None):

        self.nsim += 1
        sim = pd.Series(sim.values,
                name = 'sim_{0:02d}'.format(self.nsim),
                index=sim.index)

        if sim_name is None:
            sim_name = sim.name
        self.sim_names.append(sim_name)

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

        datam = self.data.loc[self.idx_all, :].resample('MS', how='sum')
        datam = datam.groupby(datam.index.month).mean()
        datam.columns = [self._getname(cn) for cn in  datam.columns]

        datam.plot(ax=ax, color=sim_colors, marker='o', lw=3)

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
            name = self._getname(cn)
            ax.plot(ff, value, '+-', label=name,
                markersize=6,
                color=sim_colors[icol], lw=1)
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

        dataf = data.loc[idx, :]
        dataf.columns = [self._getname(cn) for cn in  data.columns]

        dataf.plot(ax=ax, color=sim_colors, lw=2, \
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
        datay.columns = [self._getname(cn) for cn in  datay.columns]

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
        datab.plot(ax=ax, kind='bar', color=sim_colors, edgecolor='none')

        ax.set_ylabel('Average flow')
        ax.set_title('({0}) Water Balance'.format(ax_letter))
        ax.grid()


    def savefig(self, filename, size=None):
        if size is None:
            size = (12+6*(self.nfloods/3), 15)

        self.fig.set_size_inches(size)
        self.gs.tight_layout(self.fig)
        self.fig.savefig(filename)

