import numpy as np
import pandas as pd

from datetime import datetime
from calendar import month_abbr



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from hydrodiy.plot import putils, boxplot
from  hydrodiy.stat import metrics, sutils

PITCOLORS = ['black', 'red']

MEDIANLINECOLOR = putils.COLORS10[0]

OBSLINECOLOR = putils.COLORS10[3]
OBSMARKERCOLOR = 'pink'


def pitmetrics(obs, fcst):
    ''' Compute metric data

    Parameters
    -----------
    obs : numpy.ndarray
        Observed data
    fcst : numpy.ndarray
        Forecast data

    Returns
    -----------
    alpha : float
        Alpha score
    cprss_ss : float
        CRPS skill score
    pits : numpy.ndarray
        Pit data
    '''
    # Check data
    if obs.shape[0] != fcst.shape[0]:
        raise ValueError('Expected obs and fcst to have the '+\
            'same number of rows, got {0} (obs) and {1} (fcst)'.format(\
            obs.shape[0], fcst.shape[0]))

    # Compute skill scores
    _, alpha = metrics.alpha(obs, fcst)
    alpha = alpha*100
    crps, _ = metrics.crps(obs, fcst)
    crps_ss = (1.-crps[0]/crps[3])*100

    # Compute pits
    pits = metrics.pit(obs, fcst)

    return alpha, crps_ss, pits


def pitplot(pits, alpha, crps_ss, ax=None, labelaxis=True, transp=0.4):
    ''' Draw a pit plot

    Parameters
    -----------
    pits : numpy.ndarray
        Pit data
    alpha : float
        Alpha score
    cprss_ss : float
        CRPS skill score
    ax : matplotlib.Axes
        Matplotlib ax to draw on
    labelaxis : bool
        Show labels on axis
    transp : float
        Ax transparency
    '''

    # Get axis
    if ax is None:
        ax = plt.gca()

    # Draw pits
    color = PITCOLORS[alpha<5]
    pp = sutils.ppos(len(pits))
    ax.plot(np.sort(pits), pp, 'o-', \
        markersize=5, \
        color=color)

    # Decorate
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    putils.line(ax, 1, 1, 0, 0 , 'k:', lw=0.5)

    ax.text(0.95, 0.05, 'A {0:0.0f}%\n C {1:0.1f}%'.format(\
            alpha, crps_ss), \
            va='bottom', ha='right', fontsize=15, color=color)

    if labelaxis:
        ax.set_xlabel('PIT [-]')
        ax.set_ylabel('Empirical CDF [-]')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.patch.set_alpha(transp)



class MonthlyEnsplot(object):

    def __init__(self, obs, fcst, fcdates, fig=None, \
        ylabel='Flow [GL/month]'):
        ''' Object to draw monthly ensemble forecasts

        Parameters
        -----------
        obs : numpy.ndarray
            Observed data
        fcst : numpy.ndarray
            Forecast data
        fcdates : numpy.ndarray
            Forecast target month (datetime objects)
        fig : matplotlib.Figure
            Matplotlib figure to draw on
        ylabel : str
            Label of y axis

        Examples
        -----------
        >>> obs = np.random.uniform(0, 1, 100)
        >>> ens = np.random.uniform(0, 1, (100, 50))
        >>> ep = MonthlyEnsplot(obs, ens)
        >>> # Create a yearly ensemble plot
        >>> ep.yearplot()

        '''

        # Check inputs
        obs = np.atleast_1d(obs)
        fcst = np.atleast_2d(fcst)
        fcdates = pd.DatetimeIndex(fcdates)

        if obs.shape[0] != fcst.shape[0]:
            raise ValueError('Expected obs and fcst to have the '+\
                'same number of rows, got {0} (obs) and {1} (fcst)'.format(\
                obs.shape[0], fcst.shape[0]))

        if obs.shape[0] != fcdates.shape[0]:
            raise ValueError('Expected obs and fcdates to have the '+\
                'same number of rows, got {0} (obs) and {1} (fcdates)'.format(\
                obs.shape[0], fcdates.shape[0]))

        self.fcdates = fcdates
        self.obs = obs
        self.fcst = fcst

        # Config data
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig

        self.ylabel = ylabel

        self.gridspec = None


    def getdata(self, month):
        ''' Select monthly data

        Parameters
        -----------
        month : int
            Month to select data

        Return
        -----------
        qo : numpy.ndarray
            Observed data
        qf : numpy.ndarray
            Forecast data
        '''
        # Select data
        if month>0:
            idx = self.fcdates.month == month
        else:
            idx = np.ones(len(self.fcdates)).astype(bool)

        idx = idx & pd.notnull(self.obs)

        qo = np.ascontiguousarray(self.obs[idx])
        qf = np.ascontiguousarray(self.fcst[idx])
        dates = self.fcdates[idx]

        return qo, qf, dates



    def monthplot(self, month, pit=False, ax=None, xgrid=True, medline=True, \
                ymin=None, ymax=None):
        ''' Draw ensemble forecasts for a specific month

        Parameters
        -----------
        month : int
            Month used to select data
        pit : bool
            Insert pit plot or not
        xgrid : bool
            Draw x grid
        medline : bool
            Draw a line for median forecast
        ymin : float
            Minimum y axis lim
        ymax : float
            Maximum y axis lim
        '''

        # Select data
        qobs, ens, dates = self.getdata(month)
        nval = len(qobs)

        # Get axis
        if ax is None:
            ax = plt.gca()

        # Forecast boxplot
        bp = boxplot.Boxplot(ens.T, style='narrow')
        bp.median.size = 3.5
        bp.draw(ax=ax)

        qmed = bp.stats.loc['50.0%', :].values
        x = np.arange(nval)
        if medline:
            ax.plot(x, qmed, '-', linewidth=1, \
                color=MEDIANLINECOLOR, \
                markeredgecolor=MEDIANLINECOLOR)

        ax.plot(x, qobs, '-o', linewidth=2, \
                color=OBSLINECOLOR, \
                markeredgecolor=OBSLINECOLOR, \
                markerfacecolor=OBSMARKERCOLOR, \
                label='Obs')

        # Decorate
        if ymin is None:
            ymin = qobs[qobs>0].min()
        if ymax is None:
            ymax = qobs.max() * 1.2

        ax.set_ylim((ymin, ymax))
        ax.set_ylabel(self.ylabel)

        if month == 0:
            title = 'All months'
            idx = (dates.month == 1) | (dates.month == 6)
            ax.set_xticks(x[idx])
            ax.set_xticklabels(\
                [datetime.strftime(d, format='%b\n%y') for d in dates[idx]])
            ax.legend(loc=1)
        else:
            title = month_abbr[month]
            ax.set_xticks(x[::3])
            ax.set_xticklabels(dates.year[::3])

        if xgrid:
            ax.grid(axis='x')

        ax.set_title(title)

        if pit:
            alpha, crps_ss, pits = pitmetrics(qobs, ens)
            axi = inset_axes(ax, width='30%', height='30%', loc=1)
            pitplot(pits, alpha, crps_ss, ax=axi, \
                labelaxis=False)


    def yearplot(self):
        ''' Draw a figure with forecast data for all months '''

        # Initialise matplotlib objects
        gs = GridSpec(nrows=4, ncols=4)

        # Loop through months
        for month in range(13):
            if month == 0:
                ax = self.fig.add_subplot(gs[-1, :])
            else:
                ax = self.fig.add_subplot(gs[(month-1)%3, (month-1)//3])

            # Draw monthly plot
            self.monthplot(month, pit=month>0, ax=ax, xgrid=month==0, \
                medline=False)

        self.gridspec = gs


    def savefig(self, filename, figsize=(25, 18)):
        ''' Save figure at the right resolution '''

        if self.gridspec is None:
            raise ValueError('Grispec object is None, run yearplot first')

        self.fig.set_size_inches(figsize)
        self.gridspec.tight_layout(self.fig)

        self.fig.savefig(filename)
