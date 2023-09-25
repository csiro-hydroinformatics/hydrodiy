# Code modified from matplotlib demo
# https://matplotlib.org/stable/gallery/statistics/customized_violin.html

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from hydrodiy.plot import putils
from hydrodiy.plot.boxplot import compute_percentiles, BoxplotItem, COLORS

COVERAGE_CENTER = 50
COVERAGE_EXTREMES = 100

VIOLIN_WIDTH = 0.7

class ViolinplotError(Exception):
    pass

class Violin(object):
    """ Object allowing to draw violinplots """

    def __init__(self, data,
                show_text=True, \
                linewidth=2, \
                number_format="0.2f", \
                col_ref_median=COLORS[3], \
                col_ref_others=COLORS[0], \
                npoints_kde=None):
        """ Draw boxplots with labels and defined colors

        Parameters
        -----------
        data : pandas.Series, pandas.DataFrame, numpy.ndarray
            Data to be plotted
        show_text : bool
            Display summary statistics values
        linewidth : int
            Plot lines width in points
        number_format : str
            Number format for printing
        col_ref_median : str
            Color of median line
        col_ref_others : str
            Color of other elements
        npoints_kde: int
            Number points used in kde density estimation
        """
        # Check input data
        try:
            data = pd.DataFrame(data).astype(np.float64)
        except Exception as err:
            raise ViolinplotError("Failed to convert data to float"+\
                    f" dataframe: {err}")

        # initialise objects
        self.npoints_kde = 2*len(data) if npoints_kde is None else int(npoints_kde)
        self._ax = None
        self._data = data
        self._kde_x = None
        self._kde_y = None

        # Configure objects
        self.show_text = show_text
        self.linewidth = linewidth
        self.number_format = number_format
        self.col_ref_median = col_ref_median
        self.col_ref_others = col_ref_others
        self.set_items()

        # Compute violin stats
        self._compute()


    @property
    def data(self):
        """ Returns the violin data """
        return self._data

    @property
    def ax(self):
        """ Returns the boxplot axe """
        return self._ax

    @property
    def kde_x(self):
        """ Returns the kde x axis """
        return self._kde_x

    @property
    def kde_y(self):
        """ Returns the kde y axis """
        return self._kde_y


    def set_items(self):
        show_text = self.show_text
        linewidth = self.linewidth
        number_format = self.number_format
        col_ref_median = self.col_ref_median
        col_ref_others = self.col_ref_others

        self.median = BoxplotItem(linecolor=col_ref_median, \
                    fontcolor=col_ref_median,\
                    fontsize=9, \
                    fontweight="bold", \
                    marker="none",\
                    linewidth=linewidth, \
                    ha="center", va="bottom", \
                    number_format=number_format, \
                    show_text=show_text)


        col_light = putils.darken_or_lighten(col_ref_others, -0.5)
        col_superlight = putils.darken_or_lighten(col_ref_others, -1)

        self.extremes = BoxplotItem(linecolor="none", \
                            fontcolor=col_ref_others, \
                            facecolor=col_superlight, \
                            linewidth=linewidth, \
                            hatch="///", \
                            ha="center", va="bottom", \
                            number_format=number_format)

        self.center = BoxplotItem(linecolor=col_ref_others, \
                        fontcolor=col_ref_others, \
                        facecolor=col_light, \
                        width=0.7, \
                        number_format=number_format, \
                        fontsize=8, \
                        linewidth=linewidth, \
                        ha="center", va="bottom", \
                        show_text=False)


    def _compute(self):
        """ Compute stats """
        data = self._data

        # Compute stats
        self.stat_median = data.median()

        cpp1, cpp2 = compute_percentiles(COVERAGE_CENTER)
        clow = data.quantile(cpp1/100)
        self.stat_center_low = clow

        chigh = data.quantile(cpp2/100)
        self.stat_center_high = chigh

        epp1, epp2 = compute_percentiles(COVERAGE_EXTREMES)
        elow = data.quantile(epp1/100)
        self.stat_extremes_low = elow
        ehigh = data.quantile(epp2/100)
        self.stat_extremes_high = ehigh

        # initialise
        npts = self.npoints_kde
        kde_x = pd.DataFrame(np.nan, columns=data.columns, \
                                index=np.arange(npts))
        kde_y = kde_x.copy()

        # Compute kde
        for cn, se in data.items():
            kernel = gaussian_kde(se.values)

            # blend regular spacing and ecdf spacing
            x = np.linspace(elow[cn]-1, ehigh[cn]+1, (npts-len(se)))
            err = 1e-6*np.random.uniform(-1, 1, len(se))
            x = np.sort(np.concatenate([x, se.values+err]))
            y = kernel(x)
            y = (y-y.min())/(y.max()-y.min())
            kde_x.loc[:, cn] = x
            kde_y.loc[:, cn] = y

        self._kde_x = kde_x
        self._kde_y = kde_y


    def draw(self, ax=None, logscale=False):
        """ Draw the boxplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the boxplot on
        logscale : bool
            Use y axis log scale or not
        """
        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        ax = self._ax
        kde_x, kde_y = self.kde_x, self.kde_y
        ncols = kde_x.shape[1]
        vw = VIOLIN_WIDTH

        self.elements = {}
        for i, colname in enumerate(kde_x.columns):
            # Get kde data
            x = kde_x.loc[:, colname]
            y = kde_y.loc[:, colname]

            # initialise boxplot elements
            colelement = {}

            # Draw median
            med = self.stat_median[colname]
            vm = [med]*2
            u0 = np.interp(med, x, y)
            um = [i-u0*vw/2, i+u0*vw/2]
            item = self.median
            ax.plot(um, vm, lw=item.linewidth, \
                color=item.linecolor, \
                alpha=item.alpha)

            colelement["median-line"] = ax.get_lines()[-1]

            # Draw extremes
            item = self.extremes
            ix = (x>=self.stat_extremes_low[colname]) & \
                (x<=self.stat_extremes_high[colname])

            uu1, uu2 = i-y[ix]*vw/2, i+y[ix]*vw/2
            vv = x[ix]
            uc = np.concatenate([uu1, uu2[::-1], [uu1.iloc[0]]])
            vc = np.concatenate([vv, vv[::-1], [vv.iloc[0]]])
            epoly = Polygon(np.column_stack([uc, vc]), \
                                edgecolor="none", \
                                facecolor=item.facecolor, \
                                linewidth=item.linewidth, \
                                hatch=None if item.hatch=="none" else item.hatch , \
                                alpha=item.alpha)
            ax.add_patch(epoly)
            n = "extreme-polygon"
            colelement[n] = epoly

            # Draw center
            item = self.center
            ix = (x>=self.stat_center_low[colname]) & \
                    (x<=self.stat_center_high[colname])
            uu1, uu2 = i-y[ix]*vw/2, i+y[ix]*vw/2
            vv = x[ix]
            uc = np.concatenate([uu1, uu2[::-1], [uu1.iloc[0]]])
            vc = np.concatenate([vv, vv[::-1], [vv.iloc[0]]])
            cpoly = Polygon(np.column_stack([uc, vc]), \
                                edgecolor=item.linecolor, \
                                facecolor=item.facecolor, \
                                linewidth=item.linewidth, \
                                hatch=None if item.hatch=="none" else item.hatch , \
                                alpha=item.alpha)
            ax.add_patch(cpoly)
            n = "center-polygon"
            colelement[n] = cpoly

            # Text
            for statname in ["median", "centerlow", "centerhigh"]:
                item = self.median if statname == "median" else self.center
                if statname == "median":
                    item = self.median
                    value = self.stat_median[colname]
                elif statname == "centerlow":
                    item = self.center
                    value = self.stat_center_low[colname]
                elif statname == "centerhigh":
                    item = self.center
                    value = self.stat_center_high[colname]

                if item.show_text and not np.isnan(value):
                    if item.ha == "left":
                        valuetext = f" {value:{item.number_format}}"
                        xshift = vw/2
                    else:
                        valuetext = f"{value:{item.number_format}}"
                        xshift = 0

                    colelement[statname+"-text"] = \
                        ax.text(i+xshift, \
                            value, \
                            valuetext, \
                            fontweight=item.fontweight, \
                            fontsize=item.fontsize, \
                            color=item.fontcolor, \
                            va=item.va, ha=item.ha, \
                            alpha=item.alpha)



            # Store
            self.elements[colname] = colelement

        xt = np.arange(ncols)
        ax.set_xticks(xt)
        ax.set_xticklabels(kde_x.columns)
