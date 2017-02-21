from datetime import datetime
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from matplotlib import lines

COLORS = putils.COLORS10

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


class BoxplotItem(object):
    ''' Element of the boxplot graph '''

    def __init__(self, \
        linestyle='-', \
        linewidth=3, \
        linecolor='k', \
        facecolor='none', \
        va='center', \
        ha='left', \
        fontcolor='k', \
        textformat='%0.1f', \
        fontsize= 10, \
        marker='o', \
        markercolor='k', \
        showline=True, \
        showtext=True):

        # Attributes controlled by getters/setters
        self._linestyle = None
        self._va = None
        self._ha = None
        self._marker = None
        self._linewidth = None
        self._fontsize = None

        # Set attributes
        self.linestyle = linestyle
        self.va = va
        self.ha = ha
        self.marker = marker

        self.linewidth = linewidth
        self.fontsize = fontsize

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
        if not value in lines.lineStyles.keys():
            raise ValueError('Linestyle {0} not accepted'.format(value))
        self._linestyle = value


    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        if not value in ['top', 'center', 'bottom', 'baseline']:
            raise ValueError('va {0} not accepted'.format(value))
        self._va = value


    @property
    def ha(self):
        return self._ha

    @ha.setter
    def ha(self, value):
        if not value in ['left', 'center', 'right']:
            raise ValueError('va {0} not accepted'.format(value))
        self._ha = value


    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, value):
        if not value in ['o', '.', '*', '+']:
            raise ValueError('va {0} not accepted'.format(value))
        self._marker = value


    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        try:
            value = np.float64(value)
        except:
            raise ValueError('Failed to convert {0} to float'.format(value))
        self._linewidth = value


    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value):
        try:
            value = np.float64(value)
        except:
            raise ValueError('Failed to convert {0} to float'.format(value))
        self._fontsize = value





class Boxplot(object):

    def __init__(self, data, ax=None, by=None, default_width=0.7, \
                width_from_count=False,
                whiskers_coverage=80.):
        ''' Draw boxplots with labels and defined colors

        Parameters
        -----------
        data : pandas.Series, pandas.DataFrame, numpy.ndarray
            Data to be plotted
        ax : matplotlib.axes
            Axe to draw the boxplot on
        by : pandas.Series
            Categories used to split data
        default_width : float
            boxplot width
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

        self._default_width = default_width

        self._width_from_count = width_from_count

        # Box plot itements
        self.median = BoxplotItem(linecolor=COLORS[3], \
                        fontcolor=COLORS[3], fontsize=9)

        self.whiskers = BoxplotItem(linecolor=COLORS[0], \
                            linewidth=2)

        self.minmax = BoxplotItem(markercolor=COLORS[0], \
                            showline=False)

        self.box = BoxplotItem(linecolor=COLORS[0], \
                        fontcolor=COLORS[0], fontsize=8)

        self.count = BoxplotItem(fontsize=7, \
                        fontcolor='grey', textformat='%d')

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
        default_width = self._default_width
        ncols = stats.shape[1]

        qq1, qq2 = whiskers_percentiles(self.whiskerss_coverage)
        qq1txt = '{0:0.1f}%'.format(qq1)
        qq2txt = '{0:0.1f}%'.format(qq2)

        # Boxplot widths
        if self._width_from_count:
            cnt = stats.loc['count', :].values
            widths = cnt/cnt.max()*default_width
        else:
            widths = np.ones(ncols)*default_width

        for i, cn in enumerate(stats.columns):

            # Draw median
            w = widths[i]
            x = [i-w/2, i+w/2]
            med = stats.loc['50.0%', cn]
            y = [med] * 2
            valid_med = np.all(~np.isnan(med))

            item = self.median
            if item.showline and valid_med:
                ax.plot(x, y, lw=item.linewidth,
                    color=item.linecolor)

            if item.showtext and valid_med:
                formatter = item.textformat
                xshift = 0
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = w/2

                medtext = formatter % med
                ax.text(i+xshift, med, medtext, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=item.va, ha=item.ha)

            # Draw boxes
            x = [i-w/2, i+w/2, i+w/2, i-w/2, i-w/2]
            q1 = stats.loc['25.0%', cn]
            q2 = stats.loc['75.0%', cn]

            # Skip missing data
            valid_q1 = np.all(~np.isnan(q1))
            valid_q2 = np.all(~np.isnan(q2))
            if not valid_q1 or not valid_q2:
                continue

            y =  [q1, q1, q2, q2 ,q1]
            item = self.box
            if item.showline:
                ax.plot(x, y, lw=item.linewidth,
                    color=item.linecolor)

            # Draw whiskers and caps
            item = self.whiskers
            if item.showline:
                for cc in [[qq1txt, '25.0%'], [qq2txt, '75.0%']]:
                    q1 = stats.loc[cc[0], cn]
                    q2 = stats.loc[cc[1], cn]

                    x = [i]*2
                    y = [q1, q2]

                    ax.plot(x, y, lw=item.linewidth,
                        color=item.linecolor)

                    x = [i-w/5, i+w/5]
                    y = [q1]*2
                    ax.plot(x, y, lw=item.linewidth,
                        color=item.linecolor)

            # quartile values
            item = self.box
            if item.showtext:
                formatter = item.textformat
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = w/2
                elif item.ha == 'center':
                    xshift = 0

                for value in [stats.loc[qq, cn] for qq in ['25.0%', '75.0%']]:
                    valuetext = formatter % value
                    ax.text(i+xshift, value, valuetext, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=item.va, ha=item.ha)

            # min / max
            item = self.minmax
            if item.showline:
                x = [i]*2
                y = stats.loc[['min', 'max'], cn]
                ax.plot(x, y, item.marker, color=item.markercolor)


        # X tick labels
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(stats.columns)

        if not self._by is None:
            ax.set_xlabel(self._by.name)

        # Y axis log scale
        if logscale:
            ax.set_yscale('log', nonposy='clip')

        # X/Y limits
        ax.set_xlim((-default_width, ncols-1+default_width))

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
        default_width = self._default_width

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

