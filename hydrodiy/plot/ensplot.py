import numpy as np
from scipy import linalg
import pandas as pd

from datetime import datetime
from calendar import month_abbr

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from hydrodiy import HAS_C_STAT_MODULE
from hydrodiy.plot import putils, boxplot
from  hydrodiy.stat import metrics, sutils

PITCOLORS = ["dimgrey", "darkmagenta"]

SIMLINECOLOR = "midnightblue"
SIMMARKERCOLOR = "aqua"

OBSLINECOLOR = "darkred"
OBSMARKERCOLOR = "tomato"


def ensmetrics(obs, fcst, random_pit=True, stat="median"):
    """ Compute metric data

    Parameters
    -----------
    obs : numpy.ndarray
        Observed data
    fcst : numpy.ndarray
        Forecast data
    random_pit : bool
        Randomise pit computation
    stat : str
        Use mean or median for R2 computation

    Returns
    -----------
    alpha : float
        Alpha score
    cprss_ss : float
        CRPS skill score
    pits : numpy.ndarray
        Pit data
    is_sudo : numpy.ndarray
        Boolean telling if the pit value is sudo or not (i.e. censored)
    R2 : float
        Spearman correlation between obs and median or mean
        forecast
    bias : float
        Bias obs and median or mean forecast
    """
    # Check data
    if obs.shape[0] != fcst.shape[0]:
        raise ValueError("Expected obs and fcst to have the "+\
            "same number of rows, got {0} (obs) and {1} (fcst)".format(\
            obs.shape[0], fcst.shape[0]))

    if not HAS_C_STAT_MODULE:
        vnan = np.nan*np.zeros(len(obs))
        vbool = np.zeros(len(obs)).astype(bool)
        return np.nan, np.nan, vnan, vbool, np.nan, np.nan

    # Compute skill scores
    _, alpha, _ = metrics.alpha(obs, fcst)
    alpha = alpha*100
    crps, _ = metrics.crps(obs, fcst)
    crps_ss = (1.-crps[0]/crps[3])*100

    # Compute pits
    pits, is_sudo = metrics.pit(obs, fcst, random=random_pit)

    # Correlation
    R2 = metrics.corr(obs, fcst, stat=stat, type="Spearman")

    # Bias
    if stat == "mean":
        sim = np.nanmean(fcst, axis=1)
    else:
        sim = np.nanmedian(fcst, axis=1)

    bias = metrics.bias(obs, sim)

    return alpha, crps_ss, pits, is_sudo, R2, bias


def pitplot(pits, is_sudo, alpha, crps_ss, bias, ax=None, \
                labelaxis=True, transp=0.4, sudo_threshold=10):
    """ Draw a pit plot

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
    bias : float
        Forecast bias
    ax : matplotlib.Axes
        Matplotlib ax to draw on
    labelaxis : bool
        Show labels on axis
    transp : float
        Ax transparency
    sudo_threshold : int
        Percentage threshold to flag sudo pit
    """

    # Get axis
    if ax is None:
        ax = plt.gca()

    # Draw pits
    nval = len(pits)
    pp = sutils.ppos(nval)
    kk = np.argsort(pits)
    spits = pits[kk]
    ssudo = is_sudo[kk]

    color = PITCOLORS[alpha<5 \
                or np.sum(ssudo)>float(sudo_threshold)*nval/100]

    ax.plot(spits, pp, "-", color=color)

    ax.plot(spits[~ssudo], pp[~ssudo], "o", \
        markersize=5,  color=color, alpha=0.7, \
        markeredgecolor=color, markerfacecolor=color)

    prc_sudo = 100*float(np.sum(is_sudo))/len(pits)
    if prc_sudo > 0:
        ax.plot(spits[ssudo], pp[ssudo], "o", \
            markersize=5,  alpha=0.9, \
            markeredgecolor=color, markerfacecolor="w")

        ax.text(0.05, 0.95, "SP {prc_sudo:0.0f}%", \
                va="top", ha="left", fontsize=12, color=color)

    # Decorate
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    putils.line(ax, 1, 1, 0, 0 , "k:", lw=0.5)

    ax.text(0.95, 0.05, f"A {alpha:0.0f}%\n C"+\
            f" {crps_ss:0.1f}%\n B {bias:0.1f}", \
            va="bottom", ha="right", fontsize=10, color=color)

    if labelaxis:
        ax.set_xlabel("PIT [-]")
        ax.set_ylabel("Empirical CDF [-]")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.patch.set_alpha(transp)



def tsplot(obs, fcst, ax=None, \
            loc_pit=1, loc_scatter=2, loc_legend=-1, \
            line="mean", sudo_threshold=10, \
            random_pit=True):
    """ Draw ensemble forecasts timeseries
    Parameters
    -----------
    obs : numpy.ndarray
        Observed data
    fcst : numpy.ndarray
        Forecast data
    loc_pit : int
        Position of pit plot:
        * upper right	1
        * upper left	2
        * lower left	3
        * lower right	4
        * right         5
        * center left	6
        * center right	7
        * lower center	8
        * upper center	9
        * center	10
        * not shown     -1
    loc_scatter : int
        Position of scatter plot (same options than loc_pit)
    loc_legend : int
        Position of legend (same options than loc_pit)
    xgrid : bool
        Draw x grid
    line : str
        Draw a line for mean (line=mean) or median (line=median)
    sudo_threshold : int
        Percent threshold for marking sudo pits as suspicious
    random_pit : bool
        Randomise pit computation


    Returns
    -----------
    out : pandas.core.series.Series
        Aggregated time series
    x : numpy.ndarray
        Abscissae data used in plots
    alpha : float
        Alpha score (see hydrodiy.stat.metrics.alpha)
    crps_ss : float
        CRPS skill score (see hydrodiy.stat.metrics.crps)
    R2 : float
        Correlation coefficient (see hydrodiy.stat.metrics.corr)
    bias : float
        Bias (see hydrodiy.stat.metrics.bias)
   """
    # Check inputs
    nval = len(obs)

    if not line in ["mean", "median"]:
        raise ValueError("Expected line option in [mean/median], got "+line)

    if loc_scatter == loc_legend and loc_scatter >= 0:
        raise ValueError("Cannot show scatter and legend at the same place")

    if loc_scatter == loc_pit and loc_scatter >= 0:
        raise ValueError("Cannot show scatter and pit at the same place")

    if loc_pit == loc_legend and loc_pit >= 0:
        raise ValueError("Cannot show pit and legend at the same place")

    for pos in [loc_pit, loc_scatter, loc_legend]:
        if pos >= 0 and not pos in range(1, 11):
            raise ValueError("Expected location in [1, 11], got {0}".format(\
                                pos))

    # Get axis
    if ax is None:
        ax = plt.gca()

    # Forecast boxplot
    bp = boxplot.Boxplot(fcst.T, style="narrow")
    bp.median.marker = "none"
    bp.draw(ax=ax)

    # .. forecast legend
    if loc_legend >= 0:
        prc = (100.-bp.whiskers_coverage)/2
        label = "Forc. {0:0.0f}%-{1:0.0f}%".format(prc, 100-prc)
        ax.plot([], [], "-", linewidth=6, \
            alpha=bp.whiskers.alpha, \
            color=bp.whiskers.facecolor, label=label)

        prc = (100.-bp.box_coverage)/2
        label = "Forc. {0:0.0f}%-{1:0.0f}%".format(prc, 100-prc)
        ax.plot([], [], "-", linewidth=6, \
            alpha=bp.box.alpha, \
            color=bp.box.facecolor, label=label)

    # Mean and median forecast
    if line == "mean":
        qline = bp.stats.loc["mean", :].values
    else:
        qline = bp.stats.loc["50.0%", :].values

    x = np.arange(nval)
    ax.plot(x, qline, "o-", linewidth=1, \
        color=SIMLINECOLOR, \
        markerfacecolor=SIMMARKERCOLOR, \
        markeredgecolor=SIMLINECOLOR, \
        label="Forc. "+line)

    # obs data
    ax.plot(x, obs, "-o", linewidth=2, \
            color=OBSLINECOLOR, \
            markeredgecolor=OBSLINECOLOR, \
            markerfacecolor=OBSMARKERCOLOR, \
            label="Obs")

    # performance metrics
    alpha, crps_ss, pits, is_sudo, R2, bias = ensmetrics(obs, fcst, \
                                            random_pit, line)

    # Text options
    box_config = {"facecolor":"w", "edgecolor":"none", \
                        "boxstyle": "round", \
                        "alpha":0.9, "pad":0.05}
    txtcolor = PITCOLORS[0]

    # Draw figure
    if loc_pit >=0:
        axi = inset_axes(ax, width="30%", height="30%", \
                    loc=loc_pit)
        pitplot(pits, is_sudo, alpha, crps_ss, bias, ax=axi, \
            labelaxis=False, sudo_threshold=sudo_threshold)

        t = axi.text(0.05, 0.95, "PIT (ecdf)", va="top", \
                            color=txtcolor, fontsize=12, \
                            transform=axi.transAxes)
        t.set_bbox(box_config)


    if loc_scatter >= 0:
        axi2 = inset_axes(ax, width="30%", height="30%", \
                            loc=loc_scatter)
        axi2.plot(qline, obs, "o", markeredgecolor=txtcolor, \
                            markerfacecolor="w", markersize=4)

        # Create regression line
        idx = ~np.isnan(obs) & ~np.isnan(qline)
        theta, _, _, _ = linalg.lstsq(np.column_stack([qline[idx]*0+1, \
                                                    qline[idx]]), obs[idx])
        putils.line(axi2, 1, theta[1], 0, theta[0], "--", \
                            color=txtcolor, lw=1)

        t = axi2.text(0.95, 0.05, "Forc "+line.title(), ha="right", \
                            color=txtcolor, fontsize=12, \
                            transform=axi2.transAxes)
        t.set_bbox(box_config)

        t = axi2.text(0.05, 0.95, "Obs", va="top", \
                            color=txtcolor, fontsize=12, \
                            transform=axi2.transAxes)
        t.set_bbox(box_config)

        t = axi2.text(0.95, 0.95, "R2 {0:0.1f}".format(R2), \
                            va="top", ha="right", \
                            color=txtcolor, fontsize=10, \
                            transform=axi2.transAxes)
        t.set_bbox(box_config)

        axi2.set_xticks([])
        axi2.set_yticks([])
        axi2.patch.set_alpha(0.6)

    if loc_legend >= 0:
        ax.legend(loc=loc_legend, framealpha=0.6, ncol=4)

    return x, alpha, crps_ss, R2, bias



class MonthlyEnsplot(object):
    """ Object to show a summary of monthly ensemble simulations """

    def __init__(self, obs, fcst, fcdates, fig=None, \
        ylabel="Flow [GL/month]", random_pit=True, line="mean", \
        negnan=True):
        """ Object to draw monthly ensemble forecasts

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
        negnan : bool
            Set negative values to NaN

        Examples
        -----------
        >>> obs = np.random.uniform(0, 1, 100)
        >>> ens = np.random.uniform(0, 1, (100, 50))
        >>> ep = MonthlyEnsplot(obs, ens)
        >>> # Create a yearly ensemble plot
        >>> ep.overviewplot()

        """

        # Check inputs
        obs = np.atleast_1d(obs)
        if negnan:
            obs[obs < 0] = np.nan

        fcst = np.atleast_2d(fcst)
        if negnan:
            fcst[fcst < 0] = np.nan

        fcdates = pd.DatetimeIndex(fcdates)

        if obs.shape[0] != fcst.shape[0]:
            raise ValueError("Expected obs and fcst to have the "+\
                "same number of rows, got {0} (obs) and {1} (fcst)".format(\
                obs.shape[0], fcst.shape[0]))

        if obs.shape[0] != fcdates.shape[0]:
            raise ValueError(("Expected obs and fcdates to have the "+\
                "same number of rows, got {0} (obs) and"+\
                    " {1} (fcdates)").format(obs.shape[0], fcdates.shape[0]))

        self.fcdates = fcdates
        self.obs = obs
        self.fcst = fcst

        if not line in ["mean", "median"]:
            raise ValueError("Expected line option in [mean/median], got "\
                                +line)
        self.line = line

        # Config data
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig

        self.ylabel = ylabel

        self.gridspec = None

        self.random_pit = random_pit


    def getdata(self, month):
        """ Select monthly data

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
        """
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


    def monthplot(self, month, ax, loc_pit=1, \
                loc_scatter=2, loc_legend=-1):
        """ Draw monthly plot """

        # Select data
        obs, fcst, fcdates = self.getdata(month)

        # Overrides options for all month plot
        if month == 0:
            loc_scatter = -1
            loc_pit = -1
            loc_legend = 2

        # Draw ts plot
        x, alpha, crps_ss, R2, bias = tsplot(obs, fcst, ax, \
                    loc_pit = loc_pit, \
                    loc_scatter = loc_scatter, \
                    loc_legend = loc_legend, \
                    line = self.line, \
                    random_pit=self.random_pit)

        # Decorate
        if month == 0:
            title = "All months"
            idx = (fcdates.month == 1) | (fcdates.month == 7)
            ax.set_xticks(x[idx])
            ax.set_xticklabels(\
                [datetime.strftime(d, format="%b\n%y") for d in fcdates[idx]])
        else:
            title = month_abbr[month]
            ax.set_xticks(x[::3])
            ax.set_xticklabels(fcdates.year[::3])

        ax.set_title(title)

        # Min/max lims
        idx = obs > 0
        if np.sum(idx) > 0:
            ymin = obs[idx].min()
        else:
            ymin = 0.

        omax = obs.max()
        if omax > 0:
            ymax = obs.max() * 1.2
        else:
            ymax = np.nanmax(fcst)

        ax.set_ylim((ymin, ymax))

        ax.set_ylabel(self.ylabel)

        # Return performance metrics
        perf = {\
            "alpha": alpha, \
            "crps_ss": crps_ss, \
            "R2": R2, \
            "bias": bias
        }

        return perf


    def overviewplot(self, show_scatter=True, show_pit=True):
        """ Draw a figure with forecast data for all months """
        # Plot options
        loc_pit = 1 if show_pit else -1
        loc_scatter = 2 if show_scatter else -1

        # Initialise matplotlib objects
        gs = GridSpec(nrows=4, ncols=4)

        # Loop through months
        perf = {}
        axs = {}
        for month in range(13):
            if month == 0:
                ax = self.fig.add_subplot(gs[-1, :])
            else:
                ax = self.fig.add_subplot(gs[(month-1)%3, (month-1)//3])

            # Draw monthly plot
            perf[month] = self.monthplot(month, ax, \
                            loc_pit = loc_pit, \
                            loc_scatter = loc_scatter)
            # Save plot
            axs[month] = ax

        self.gridspec = gs
        perf = pd.DataFrame(perf).T
        perf.index.name = "month"

        return axs, perf


    def set_overview_fig(self, title="", figsize=(26, 18)):
        """ Set figure dimensions """

        if self.gridspec is None:
            raise ValueError("Grispec object is None, run overviewplot first")

        self.fig.set_size_inches(figsize)

        if title != "":
            self.fig.suptitle(title)
            self.gridspec.tight_layout(self.fig, rect=[0, 0, 1, 0.97])
        else:
            self.gridspec.tight_layout(self.fig)

