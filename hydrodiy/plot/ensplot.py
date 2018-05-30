import numpy as np
import pandas as pd

from datetime import datetime
from calendar import month_abbr



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from hydrodiy.plot import putils, boxplot
from  hydrodiy.stat import metrics, sutils

PITCOLORS = ['dimgrey', 'darkmagenta']

SIMLINECOLOR = 'midnightblue'
SIMMARKERCOLOR = 'aqua'

OBSLINECOLOR = 'darkred'
OBSMARKERCOLOR = 'tomato'


def pitmetrics(obs, fcst, random=True):
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
    pits, is_sudo = metrics.pit(obs, fcst, random=random)

    return alpha, crps_ss, pits, is_sudo


def pitplot(pits, is_sudo, alpha, crps_ss, ax=None, labelaxis=True, \
                transp=0.4, random=False):
    ''' Draw a pit plot

    Parameters
    -----------
    pits : numpy.ndarray
        Pit data
    is_sudo : numpy.ndarray
        Boolean series containing the check if pit values are sudo pits
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
    nval = len(pits)
    pp = sutils.ppos(nval)
    kk = np.argsort(pits)
    spits = pits[kk]
    ssudo = is_sudo[kk]

    color = PITCOLORS[alpha<5 or np.sum(ssudo)>5e-2*nval]

    ax.plot(spits, pp, '-', color=color)

    ax.plot(spits[~ssudo], pp[~ssudo], 'o', \
        markersize=5,  color=color, alpha=0.7, \
        markeredgecolor=color, markerfacecolor=color)

    prc_sudo = 100*float(np.sum(is_sudo))/len(pits)
    if prc_sudo > 0:
        ax.plot(spits[ssudo], pp[ssudo], 'o', \
            markersize=5,  alpha=0.9, \
            markeredgecolor=color, markerfacecolor='w')

        ax.text(0.05, 0.95, 'SP {0:0.0f}%'.format(prc_sudo), \
                va='top', ha='left', fontsize=12, color=color)

    # Decorate
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    putils.line(ax, 1, 1, 0, 0 , 'k:', lw=0.5)

    ax.text(0.95, 0.05, 'A {0:0.0f}%\n C {1:0.1f}%'.format(\
            alpha, crps_ss), \
            va='bottom', ha='right', fontsize=12, color=color)

    if labelaxis:
        ax.set_xlabel('PIT [-]')
        ax.set_ylabel('Empirical CDF [-]')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.patch.set_alpha(transp)



def tsplot(obs, fcst, ax=None, \
            show_pit=False, show_scatter=False, \
            line='mean'):
    ''' Draw ensemble forecasts timeseries
    Parameters
    -----------
    obs : numpy.ndarray
        Observed data
    fcst : numpy.ndarray
        Forecast data
    show_pit : bool
        Insert pit plot or not
    show_scatter : bool
        Insert median scatter plot or not
    xgrid : bool
        Draw x grid
    line : str
        Draw a line for mean (line=mean) or median (line=median)
   '''
    # Check inputs
    nval = len(obs)

    if not line in ['mean', 'median']:
        raise ValueError('Expected line option in [mean/median], got '+line)

    # Get axis
    if ax is None:
        ax = plt.gca()

    # Forecast boxplot
    bp = boxplot.Boxplot(fcst.T, style='narrow')
    bp.median.marker = 'none'
    bp.draw(ax=ax)

    # Mean and median forecast
    if line == 'mean':
        qline = bp.stats.loc['mean', :].values
    else:
        qline = bp.stats.loc['50.0%', :].values

    x = np.arange(nval)
    ax.plot(x, qline, 'o-', linewidth=1, \
        color=SIMLINECOLOR, \
        markerfacecolor=SIMMARKERCOLOR, \
        markeredgecolor=SIMLINECOLOR)

    # obs data
    ax.plot(x, obs, '-o', linewidth=2, \
            color=OBSLINECOLOR, \
            markeredgecolor=OBSLINECOLOR, \
            markerfacecolor=OBSMARKERCOLOR, \
            label='Obs')

    if show_pit:
        alpha, crps_ss, pits, is_sudo = pitmetrics(obs, fcst)
        axi = inset_axes(ax, width='30%', height='30%', loc=1)
        pitplot(pits, is_sudo, alpha, crps_ss, ax=axi, \
            labelaxis=False)

    if show_scatter:
        axi2 = inset_axes(ax, width='30%', height='30%', loc=2)
        axi2.plot(qline, obs, 'o')
        putils.line(axi2, 1, 1, 0, 0, '-', linewidth=1, color='grey')

        # Create regression line
        theta, _, _, _ = np.linalg.lstsq(np.column_stack([qline*0+1, \
                                                        qline]), obs)
        putils.line(axi2, 1, theta[1], 0, theta[0], 'k--')

        axi2.text(0.95, 0.05, 'Sim', ha='right', transform=axi2.transAxes)
        axi2.text(0.05, 0.95, 'Obs', va='top', transform=axi2.transAxes)

        axi2.set_xticks([])
        axi2.set_yticks([])
        axi2.patch.set_alpha(0.3)

    return x



class MonthlyEnsplot(object):

    def __init__(self, obs, fcst, fcdates, fig=None, \
        ylabel='Flow [GL/month]', randompit=False, line='mean'):
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
        randompit : bool
            Randomise pit computation (generate pseudo-pits)
        line : str
            Line option to show mean or median

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

        if not line in ['mean', 'median']:
            raise ValueError('Expected line option in [mean/median], got '\
                                +line)
        self.line = line

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


    def monthplot(self, month, ax, show_pit=True, \
                show_scatter=True):
        ''' Draw monthly plot '''
        # Select data
        obs, fcst, fcdates = self.getdata(month)

        # Draw ts plot
        x = tsplot(obs, fcst, ax, \
                    show_pit, show_scatter, \
                    self.line)

        # Decorate
        if month == 0:
            title = 'All months'
            idx = (fcdates.month == 1) | (fcdates.month == 6)
            ax.set_xticks(x[idx])
            ax.set_xticklabels(\
                [datetime.strftime(d, format='%b\n%y') for d in fcdates[idx]])
            ax.legend(loc=1)
        else:
            title = month_abbr[month]
            ax.set_xticks(x[::3])
            ax.set_xticklabels(fcdates.year[::3])

        ax.set_title(title)

        ymin = obs[obs>0].min()
        ymax = obs.max() * 1.2
        ax.set_ylim((ymin, ymax))

        ax.set_ylabel(self.ylabel)


    def yearplot(self, show_scatter=True, show_pit=True):
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
            self.monthplot(month, ax, \
                    show_pit = month>0 and show_pit, \
                    show_scatter = month>0 and show_scatter)

        self.gridspec = gs


    def savefig(self, filename, figsize=(26, 18)):
        ''' Save figure at the right resolution '''

        if self.gridspec is None:
            raise ValueError('Grispec object is None, run yearplot first')

        self.fig.set_size_inches(figsize)
        self.gridspec.tight_layout(self.fig)

        self.fig.savefig(filename)
