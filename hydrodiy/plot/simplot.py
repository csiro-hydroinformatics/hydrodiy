import warnings

import re
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta
from itertools import cycle

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
COLOR_SCHEME = putils.COLORS10

def get_colors():
    return cycle(COLOR_SCHEME)


class Simplot(object):
    ''' Object to plot a summary of daily hydrological simulations '''

    def __init__(self, obs, sim, \
        sim_name = None,
        fig = None, \
        nfloods = 3, \
        wateryear_start =7, \
        fdc_zoom_xlim=[0., 0.05], \
        fdc_zoom_ylog=True, \
        ndays_beforepeak = 30, \
        ndays_afterpeak = 60, \
        samefloodyscale=False, \
        maxnan_annual=50, \
        maxnan_monthly=10):
        ''' Object to generate a diagnostic plot for flow simulations

        Parameters
        -----------
        obs : pandas.Series
            Times series of observed data
        sim : pandas.Series
            Times series of simulated data
        sim_name : str
            Name of simulated data to be used on the plots
        fig : matplotlib.Figure
            Figure to draw plots on
        nfloods : int
            Number of flood events to draw
        wateryear_start : int
            Month of water year start
        fdc_zoom_xlim : list
            X axis limits for the fdc zoom plot
        fdc_zoom_ylog : bool
            Log transform Y axis in fdc zoom plot
        ndays_beforepeak : int
            Number of days preceeding the peak in flood plots
        ndays_afterpeak : int
            Number of days following the peak in flood plots
        samefloodyscale : bool
            Use the same yscale for flood plots
        maxnan_annual : int
            Maximum number of nan per year
        maxnan_monthly : int
            Maximum number of nan per month
        '''

        # Properties
        self.samefloodyscale = samefloodyscale
        self.wateryear_start = wateryear_start
        self.fdc_zoom_xlim = [float(x) for x in list(fdc_zoom_xlim)]
        self.fdc_zoom_ylog = bool(fdc_zoom_ylog)

        # data
        self.idx_obs = pd.notnull(obs) & (obs >= 0)
        obs.name = 'obs'
        self.minobs = obs[self.idx_obs].min()
        self.maxobs = obs[self.idx_obs].max()
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

        # Max nan values
        self.maxnan_annual = maxnan_annual
        self.maxnan_monthly = maxnan_monthly

        # Figure to draw on
        if fig is None:
            fig = plt.figure()
        self.fig = fig

        # Grid spec
        fig_ncols = 3
        fig_nrows = 3 + (nfloods-1)//3
        self.gs = gridspec.GridSpec(fig_nrows, fig_ncols)
                #width_ratios=[1] * fig_ncols,
                #height_ratios=[0.5] * 1 + [1] * (fig_nrows-1))


    def _compute_idx_all(self):
        ''' Compute indexes where all data are available'''
        self.idx_all = pd.isnull(self.data).sum(axis=1) == 0

        if np.sum(self.idx_all) == 0:
            raise ValueError('No common data points between '+\
                    'time series')


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

            omax = obs_tmp[idx].max()
            date_max = dates[idx][obs_tmp[idx] > omax-1e-10][0]
            idx = dates >= date_max - delta(days=self.ndays_beforepeak)
            idx = idx & (dates <= date_max \
                                + delta(days=self.ndays_afterpeak))

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
        ''' Draw all simulations plots

        Returns
        -----------
        axb : matplotlib.axes.Axes
            Axes showing water balance data

        axa : matplotlib.axes.Axes
            Axes showing annual time series

        axfd : matplotlib.axes.Axes
            Axes showing flow duration curve in log-flow space (low flow)

        axfdl : matplotlib.axes.Axes
            Axes showing flow duration curve in log-frequency space (high flow)

        axs : matplotlib.axes.Axes
            Axes showing mean monthly averages (seasonal patterns)

        axf : list
            List containing the axes showing flood simulations

        '''
        # Draw water balance
        axb = plt.subplot(self.gs[0, 0])
        #self.draw_balance(axb)
        self.draw_loglog(axb)

        # Draw annual time series
        axa = plt.subplot(self.gs[0, 1:])
        self.draw_monthly(axa)

        # Draw flow duration curves
        axfd = plt.subplot(self.gs[1, 0])
        self.draw_fdc(axfd)

        axfdl = plt.subplot(self.gs[1, 1])
        self.draw_fdc(axfdl, 'd', \
                    xlim = self.fdc_zoom_xlim, \
                    ylog = self.fdc_zoom_ylog, \
                    title_postfix='- high flow zoom')

        # Draw seasonal residuals
        axs = plt.subplot(self.gs[1, 2])
        self.draw_monthlymeans(axs, 'e')

        # Same flood y scale if needed
        ylim = (max(0, self.minobs*0.5), self.maxobs*1.2)

        # Draw flood events
        axf = []
        for iflood in range(self.nfloods):
            ix = 2 + iflood//3
            iy = iflood%3
            ax = plt.subplot(self.gs[ix, iy])
            self.draw_floods(ax, iflood, letters[5+iflood], \
                        show_legend=iflood == 0)

            if self.samefloodyscale:
                ax.set_ylim(ylim)

            axf.append(ax)

        return axb, axa, axfd, axfdl, axs, axf


    def draw_monthlymeans(self, ax, ax_letter='d'):
        ''' Draw the plot of monthly residuals '''

        # Quick aggregate
        data = self.data
        months = data.index.year * 100 + data.index.month
        monthsu = pd.to_datetime(pd.np.unique(months), format='%Y%m')
        lam = lambda x: pd.Series(dutils.aggregate(months, x.values, \
                                    maxnan=self.maxnan_monthly),\
                                        index=monthsu)
        datam = self.data.apply(lam)

        # Compute monthly means
        mdatam = datam.groupby(datam.index.month).mean()
        mdatam.columns = [self._getname(cn) for cn in  mdatam.columns]

        # plot mean monthly
        lines = {}
        colors = get_colors()
        for (cn, se), color in zip(mdatam.items(), colors):
            se.plot(ax=ax, color=color, marker='o', lw=3)

        # decoration
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=2, frameon=False)

        ax.set_xlim((0, 13))
        ax.set_xticks(range(1, 13))
        ax.grid()
        ax.set_xlabel('Month')
        ax.set_ylabel('Flow')
        title = '({0}) Monthly means'.format(ax_letter)
        ax.set_title(title)

        return lines


    def draw_fdc(self, ax, ax_letter='c', xlog=False, \
                        ylog=True, xlim=[0, 1], title_postfix=''):
        ''' Draw the flow duration curve '''

        # Freqency
        data = self.data
        idx = self.idx_all
        nval = np.sum(idx)
        ff = sutils.ppos(nval)

        icol = 0
        ymin = np.inf
        lines = {}
        colors = get_colors()
        for cn, color in zip(data.columns, colors):
            value = np.sort(data.loc[idx, cn].values)[::-1]
            name = self._getname(cn)
            ax.plot(ff, value, '-', label=name,
                markersize=6,
                color=color, lw=2)
            lines[cn] = ax.get_lines()[-1]
            icol += 1

            ymincn = np.min(value[ff < xlim[1]])
            if ymincn < ymin:
                ymin = ymincn

        # Decorate
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Flow')

        title = '({0}) Flow duration curve {1}'.format(\
                            ax_letter, title_postfix)
        ax.set_title(title)

        ax.legend(loc=1, frameon=False)
        ax.grid()

        # Limits
        if xlog:
            xmin = max(xlim[0], ff[0]/10)
            ax.set_xlim([xmin, xlim[1]])
        else:
            ax.set_xlim(xlim)

        _, ymax = ax.get_ylim()
        if ylog:
            ymin = max(np.min(value[value>0])/10, ymin)

        ax.set_ylim((ymin, ymax))

        # Scales
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

        return lines


    def draw_floods(self, ax, iflood, ax_letter, yminfactor=0.5, \
                    ymaxfactor=1.2, show_legend=False):

        ''' Draw a plot for a single flood event '''
        # Select event
        data = self.data
        idx = self.flood_idx[iflood]['index']

        dataf = data.loc[idx, :]
        dataf.columns = [self._getname(cn) for cn in  data.columns]

        # Draw flood plot
        lines = {}
        colors = get_colors()
        for (cn, se), color in zip(dataf.items(), colors):
            se.plot(ax=ax, color=color, lw=2, \
                marker='o', legend=iflood==0)
            lines[cn] = ax.get_lines()[-1]

        if show_legend:
            leglines, labels = ax.get_legend_handles_labels()
            ax.legend(leglines, labels, loc=2, frameon=False)

        date_max = self.flood_idx[iflood]['date_max']
        title = r'({0}) Flood #{1} - {2:%b %Y}'.format(ax_letter, \
            iflood+1, date_max)
        ax.set_title(title)
        ax.xaxis.grid()
        ax.set_ylabel('Flow')

        return lines


    def draw_monthly(self, ax, ax_letter='b'):
        ''' Draw plot for annual time series '''

        # Compute monthly time series
        data = self.data
        months = data.index.year*100 + data.index.month

        monthsunique = [datetime(int(m/100), m-100*int(m/100), 1) \
                            for m in np.unique(months)]
        lam = lambda x: pd.Series(dutils.aggregate(months, x.values, \
                                    maxnan=self.maxnan_annual), \
                                        index=monthsunique)
        datay = data.apply(lam)

        # plot - exclude first and last year to avoid missing values
        lines = {}
        colors = get_colors()
        for (cn, se), color in zip(datay.items(), colors):
            se.plot(ax=ax, color=color, lw=2)
            lines[cn] = ax.get_lines()[-1]

        leglines, labels = ax.get_legend_handles_labels()
        ax.legend(leglines, labels, loc=2, frameon=False)

        title = '({0}) Monthly time series'.format(ax_letter)
        ax.set_title(title)
        ax.set_ylabel('Monthly flow')
        ax.xaxis.grid()

        return lines


    def draw_balance(self, ax, ax_letter='a'):
        ''' Draw bar chart with mean annual data '''
        data = self.data
        cc = [self._getname(cn) for cn in  data.columns]
        datab = pd.Series(data.loc[self.idx_all, :].mean().values,
                    index=cc)

        freq = data.index.freqstr
        freqfact = {'H':24*365.25, 'D':365.25, 'MS':12, 'ME':12}
        if freq not in freqfact:
            warnings.warn(('Frequency [{0}] not recognised, '+\
                            'assuming daily').format(freq))
            fact = freqfact['D']
        else:
            fact = freqfact[freq]

        datab = datab * fact

        # plot
        lines = {}
        colors = get_colors()
        for (cn, se), color in zip(datab.items(), colors):
            se.plot(ax=ax, kind='bar', color=color, edgecolor='none')
            lines[cn] = ax.get_lines()[-1]

        ax.set_ylabel('Mean annual')
        ax.set_title('({0}) Water Balance'.format(ax_letter))
        ax.grid()

        return lines


    def draw_loglog(self, ax, ax_letter='a'):
        ''' Draw a log-log plot of sorted sim vs obs '''
        data = self.data
        obs = np.sort(data.loc[self.idx_all, data.columns[0]])
        lines = {}
        colors = get_colors()
        for cn, color in zip(data.columns[1:], colors):
            sim = np.sort(data.loc[self.idx_all, cn])
            label = self._getname(cn)
            ax.loglog(sim, obs, label=label, color=color)
            lines[cn] = ax.get_lines()[-1]

        putils.line(ax, 1, 1, 1, 1, 'k--')
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Observation')
        ax.legend(loc=4, fontsize='x-small')
        ax.set_title('({0}) Sim vs obs'.format(ax_letter))
        ax.grid()

        return lines


    def set_size_inches(self, size=None):
        ''' Set figure size '''
        if size is None:
            size = (20, 10+5*((self.nfloods-1)/3+1))

        self.fig.set_size_inches(size)
        self.gs.tight_layout(self.fig)

        return size


    def savefig(self, filename, size=None):
        ''' Save figure to file '''
        size = self.set_size_inches(size)
        self.fig.savefig(filename)

        return size

