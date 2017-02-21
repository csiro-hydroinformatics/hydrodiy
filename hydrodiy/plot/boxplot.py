from datetime import datetime
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from matplotlib import lines
from matplotlib.patches import FancyBboxPatch, BoxStyle

COLORS = putils.COLORS10
EPS = 1e-10

def whiskers_percentiles(coverage):
    ''' Compute whiskers percentiles from coverage '''
    qq1 = float(100-coverage)/2
    qq2 = 100.-qq1
    return qq1, qq2


def boxplot_stats(data, coverage):
    ''' Compute boxplot stats '''

    idx = (~np.isnan(data)) & (~np.isinf(data))
    qq1, qq2 = whiskers_percentiles(coverage)
    nok = np.sum(idx)

    if nok>3:
        qq = [qq1, 25, 50, 75, qq2]
        prc = pd.Series(np.nanpercentile(data[idx], qq), \
                        index=['{0:0.1f}%'.format(qqq) for qqq in qq])
        prc['count'] = nok
        prc['mean'] = data[idx].mean()
        prc['max'] = data[idx].max()
        prc['min'] = data[idx].min()
    else:
        prc = pd.Series({'count': nok})
        pnames = ['{0:0.1f}%'.format(qq1), '25.0%', \
                '50.0%', '75.0%', '{0:0.1f}%'.format(qq2), \
                'min', 'max', 'mean']
        for pn in pnames:
            prc[pn] = np.nan

    return prc


def _to_float(value, mini=0., maxi=np.inf):
    try:
        value = np.float64(value)
    except:
        raise ValueError('Failed to convert {0} to float'.format(value))

    if value<mini-EPS or value>maxi+EPS:
        raise ValueError(('Expected value in [{0}, {1}], ' +\
            'got {2}').format(mini, maxi, value))

    return value


def _is_in(value, possible):
    if not value in possible:
        raise ValueError(('Expected value in {0},' + \
            ' got {1}').format(possible, value))

    return value



class BoxplotItem(object):
    ''' Element of the boxplot graph '''

    def __init__(self, \
        linestyle='-', \
        alpha=1., \
        linewidth=3, \
        linecolor='k', \
        facecolor='none', \
        width=0.2, \
        va='center', \
        ha='left', \
        fontcolor='k', \
        textformat='%0.1f', \
        fontsize=10, \
        boxstyle='square', \
        marker='o', \
        markercolor='k', \
        showline=True, \
        showtext=True):

        # Attributes controlled by getters/setters
        self._linestyle = None
        self._alpha = None
        self._va = None
        self._ha = None
        self._marker = None
        self._boxstyle = None
        self._linewidth = None
        self._fontsize = None
        self._width = None

        # Set attributes
        self.linestyle = linestyle
        self.alpha = alpha
        self.va = va
        self.ha = ha
        self.marker = marker
        self.boxstyle = boxstyle

        self.linewidth = linewidth
        self.fontsize = fontsize
        self.width = width

        self.linecolor = linecolor
        self.facecolor = facecolor
        self.fontcolor = fontcolor
        self.markercolor = markercolor

        self.textformat = textformat

        self.showline = showline
        self.showtext = showtext


    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        self._linestyle = _is_in(value, lines.lineStyles.keys())


    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        self._va = _is_in(value, ['top', 'center', \
                                        'bottom', 'baseline'])

    @property
    def ha(self):
        return self._ha

    @ha.setter
    def ha(self, value):
        self._ha = _is_in(value, ['left', 'center', \
                                        'right'])

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, value):
        self._marker = _is_in(value, ['none', \
                    'o', '.', '*', '+'])

    @property
    def boxstyle(self):
        return self._boxstyle

    @boxstyle.setter
    def boxstyle(self, value):
        self._boxstyle = _is_in(value, ['square', \
                    'round'])

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        self._linewidth = _to_float(value)


    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value):
        self._fontsize = _to_float(value)


    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = _to_float(value)


    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = _to_float(value, maxi=1.)



class Boxplot(object):

    def __init__(self, data, ax=None, \
                style='default', by=None, \
                width_from_count=False,
                whiskers_coverage=80.):
        ''' Draw boxplots with labels and defined colors

        Parameters
        -----------
        data : pandas.Series, pandas.DataFrame, numpy.ndarray
            Data to be plotted
        ax : matplotlib.axes
            Axe to draw the boxplot on
        style : str
            Boxplot style. Possible:
            * default : Standard boxplot
            * narrow : Boxplot for large amount of data
        by : pandas.Series
            Categories used to split data
        width_from_count : bool
            Use counts to define the boxplot width
        whiskers_coverage : float
            Coverage defining the percentile used to compute whiskers extent.
            Example: coverage = 80% -> 10%/90% whiskers
        '''

        # Check input data
        if not by is None:
            by = pd.Series(by)
            if by.name is None:
                by.name = 'by'

            if len(by.unique()) == 1:
                raise ValueError('by has 1 category only')

            data = pd.Series(data)
        else:
            data = pd.DataFrame(data)

        self._data = data

        self._by = by

        if whiskers_coverage <= 50.:
            raise ValueError('Whiskers coverage cannot be below 50.')
        self.whiskerss_coverage = whiskers_coverage

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        self._width_from_count = width_from_count

        # Box plot items
        if style == 'default':
            self.median = BoxplotItem(linecolor=COLORS[3], \
                            fontcolor=COLORS[3], fontsize=9, \
                            marker='none')

            self.whiskers = BoxplotItem(linecolor=COLORS[0], \
                                width=0.3, linewidth=2)

            self.box = BoxplotItem(linecolor=COLORS[0], \
                            width=0.7, fontcolor=COLORS[0], fontsize=8)

        elif style == 'narrow':
            self.median = BoxplotItem(marker='o', markercolor=COLORS[3], \
                            showline=False, \
                            showtext=False)

            self.whiskers = BoxplotItem(linecolor=COLORS[1], \
                                width=0., linewidth=5)

            self.box = BoxplotItem(linecolor=COLORS[0], \
                            alpha=0.5, facecolor=COLORS[0], \
                            width=0.6, showtext=False)

        else:
            raise ValueError('Style {0} not implemented'.format(style))

        # Items not affected by style
        self.count = BoxplotItem(fontsize=7, \
                        fontcolor='grey', textformat='%d')

        self.minmax = BoxplotItem(markercolor=COLORS[0], \
                            marker='none', \
                            showline=False)

        # Compute boxplot stats
        self._compute()


    def _compute(self):
        ''' Compute boxplot stats '''

        data = self._data
        by = self._by
        whc = self.whiskerss_coverage

        if by is None:
            self._stats = data.apply(boxplot_stats, args=(whc, ))
        else:
            stats = data.groupby(by).apply(boxplot_stats, whc)

            # Reformat to make a 2d dataframe
            stats = stats.reset_index()
            self._stats = pd.pivot_table(stats, \
                index='level_1', columns=by.name, values=stats.columns[-1])


    @property
    def stats(self):
        return self._stats


    def draw(self, logscale=False):
        ''' Draw the boxplot '''

        ax = self._ax
        stats = self._stats
        ncols = stats.shape[1]

        qq1, qq2 = whiskers_percentiles(self.whiskerss_coverage)
        qq1txt = '{0:0.1f}%'.format(qq1)
        qq2txt = '{0:0.1f}%'.format(qq2)

        # Boxplot widths
        if self._width_from_count:
            cnt = stats.loc['count', :].values
            ratio = cnt/cnt.max()
            boxwidths = ratio*self.box.width
            capwidths = ratio*self.whiskers.width
        else:
            boxwidths = np.ones(ncols)*self.box.width
            capwidths = np.ones(ncols)*self.whiskers.width

        # Loop through stats
        for i, cn in enumerate(stats.columns):
            # Widths
            bw = boxwidths[i]
            cw = capwidths[i]

            # Draw median
            x = [i-bw/2, i+bw/2]
            med = stats.loc['50.0%', cn]
            y = [med] * 2
            valid_med = np.all(~np.isnan(med))

            item = self.median
            if item.showline and valid_med:
                ax.plot(x, y, lw=item.linewidth, \
                    color=item.linecolor, \
                    alpha=item.alpha)

            if item.marker != 'none':
                ax.plot(i, med, marker=item.marker, \
                    color=item.markercolor, \
                    alpha=item.alpha)

            if item.showtext and valid_med:
                formatter = item.textformat
                xshift = 0
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = bw/2

                medtext = formatter % med
                ax.text(i+xshift, med, medtext, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=item.va, ha=item.ha, \
                        alpha=item.alpha)

            # Skip missing data
            q1 = stats.loc['25.0%', cn]
            q2 = stats.loc['75.0%', cn]

            valid_q1 = np.all(~np.isnan(q1))
            valid_q2 = np.all(~np.isnan(q2))
            if not valid_q1 or not valid_q2:
                continue

            # Draw box
            item = self.box
            if item.facecolor !='none' or item.showline:
                bbox = FancyBboxPatch([i-bw/2, q1], bw, q2-q1, \
                    boxstyle=BoxStyle(item.boxstyle, pad=0.), \
                    facecolor=item.facecolor, \
                    linewidth=item.linewidth, \
                    edgecolor=item.linecolor,
                    alpha=item.alpha)

                ax.add_patch(bbox)

            # Draw whiskers and caps
            item = self.whiskers
            if item.showline:
                for cc in [[qq1txt, '25.0%'], [qq2txt, '75.0%']]:
                    q1 = stats.loc[cc[0], cn]
                    q2 = stats.loc[cc[1], cn]

                    x = [i]*2
                    y = [q1, q2]

                    ax.plot(x, y, lw=item.linewidth,
                        color=item.linecolor, \
                        alpha=item.alpha)

                    if cw>0.:
                        x = [i-cw/5, i+cw/5]
                        y = [q1]*2
                        ax.plot(x, y, lw=item.linewidth,
                            color=item.linecolor, \
                            alpha=item.alpha)

            # quartile values
            item = self.box
            if item.showtext:
                formatter = item.textformat
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = bw/2
                elif item.ha == 'center':
                    xshift = 0

                for value in [stats.loc[qq, cn] for qq in ['25.0%', '75.0%']]:
                    valuetext = formatter % value
                    ax.text(i+xshift, value, valuetext, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=item.va, ha=item.ha, \
                        alpha=item.alpha)

            # min / max
            item = self.minmax
            if item.marker != 'none':
                x = [i]*2
                y = stats.loc[['min', 'max'], cn]
                ax.plot(x, y, item.marker, color=item.markercolor, \
                    alpha=item.alpha)

        # X tick labels
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(stats.columns)

        if not self._by is None:
            ax.set_xlabel(self._by.name)

        # Y axis log scale
        if logscale:
            ax.set_yscale('log', nonposy='clip')

        # X/Y limits
        w = np.max(boxwidths)
        ax.set_xlim((-w, ncols-1+w))

        ylim = ax.get_ylim()
        if not logscale:
            dy = ylim[1]-ylim[0]
            ylim0 = min(ylim[0], stats.loc[qq1txt, :].min()-dy*0.02)
            ylim1 = max(ylim[1], stats.loc[qq2txt, :].max()+dy*0.02)
        else:
            dy = math.log(ylim[1])-math.log(ylim[0])
            ylim0 = ylim[0]
            miniq = stats.loc[qq1txt, :].min()
            if miniq>0:
                ylim0 = min(ylim[0], math.exp(math.log(miniq)-dy*0.02))

            ylim1 = ylim[1]
            maxiq = stats.loc[qq1txt, :].max()
            if maxiq>0:
                ylim1 = max(ylim[1], math.exp(math.log(maxiq)+dy*0.02))

        ax.set_ylim((ylim0, ylim1))


    def show_count(self, ypos=0.025):
        ''' Show the counts '''

        ax = self._ax
        stats = self._stats

        va = 'bottom'
        if ypos > 0.5:
            va = 'top'

        item = self.count
        if item.showtext:
            # Get y coordinate from ax coordinates
            trans1 = ax.transAxes.transform
            trans2 = ax.transData.inverted().transform
            _, y1 = trans1((0, ypos))
            _, y = trans2((0., y1))

            formatter = '('+item.textformat+')'

            for i, cn in enumerate(stats.columns):
                cnt = formatter % stats.loc['count', cn]
                ax.text(i, y, cnt, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=va, ha='center')
        else:
            raise ValueError('showtext property for count set to False')

