import re
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from matplotlib import lines
from matplotlib.patches import FancyBboxPatch, BoxStyle
import matplotlib.pyplot as plt

COLORS = putils.COLORS10
EPS = 1e-10

class BoxplotError(Exception):
    pass

def compute_percentiles(coverage):
    ''' Compute whiskers percentiles from coverage '''
    qq1 = float(100-coverage)/2
    qq2 = 100.-qq1
    return qq1, qq2


def boxplot_stats(data, box_coverage, whiskers_coverage):
    ''' Compute boxplot stats '''

    idx = (~np.isnan(data)) & (~np.isinf(data))
    bqq1, bqq2 = compute_percentiles(box_coverage)
    wqq1, wqq2 = compute_percentiles(whiskers_coverage)
    nok = np.sum(idx)

    if nok>3:
        qq = [wqq1, bqq1, 50, bqq2, wqq2]
        prc = pd.Series(np.nanpercentile(data[idx], qq), \
                        index=['{0:0.1f}%'.format(qqq) for qqq in qq])
        prc['count'] = nok
        prc['mean'] = data[idx].mean()
        prc['max'] = data[idx].max()
        prc['min'] = data[idx].min()
    else:
        prc = pd.Series({'count': nok})
        pnames = ['{0:0.1f}%'.format(wqq1), \
                '{0:0.1f}%'.format(bqq1), \
                '50.0%', \
                '{0:0.1f}%'.format(bqq2), \
                '{0:0.1f}%'.format(wqq2), \
                'min', 'max', 'mean']
        for pn in pnames:
            prc[pn] = np.nan

    return prc


def _to_float(value, mini=0., maxi=np.inf):
    ''' Convert value to float with range checking '''
    try:
        value = np.float64(value)
    except:
        raise BoxplotError('Failed to convert {0} to float'.format(value))

    if value<mini-EPS or value>maxi+EPS:
        raise BoxplotError(('Expected value in [{0}, {1}], ' +\
            'got {2}').format(mini, maxi, value))

    return value


def _is_in(value, possible):
    ''' Check if a value is in a set of possible choices '''
    if not value in possible:
        raise BoxplotError(('Expected value in {0},' + \
            ' got {1}').format(possible, value))

    return value



class BoxplotItem(object):
    ''' Element of the boxplot graph '''

    def __init__(self, \
        linestyle='-', \
        alpha=1., \
        linewidth=2, \
        linecolor='k', \
        facecolor='none', \
        width=0.2, \
        va='center', \
        ha='left', \
        fontcolor='k', \
        textformat='%0.1f', \
        fontsize=10, \
        boxstyle='Square,pad=0', \
        marker='o', \
        markerfacecolor='none', \
        markeredgecolor='k', \
        markersize=5, \
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
        self._markersize = None
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
        self.markersize = markersize
        self.width = width

        self.linecolor = linecolor
        self.facecolor = facecolor
        self.fontcolor = fontcolor
        self.markerfacecolor = markerfacecolor
        self.markeredgecolor = markeredgecolor

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
        if not re.search('^(Square|Round)', value):
            raise BoxplotError(('Boxstyle should start either' +\
                ' by Round or Square, got {0}').format(value))
        self._boxstyle = value

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
    def markersize(self):
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        self._markersize = _to_float(value)


    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = _to_float(value, maxi=1.)



class Boxplot(object):
    ''' Object allowing to draw boxplots '''

    def __init__(self, data,
                style='default', by=None, \
                showtext=True, \
                centertext=False, \
                linewidth=2, \
                width_from_count=False,
                digitnumber=2, \
                box_coverage=50.,
                whiskers_coverage=90.):
        ''' Draw boxplots with labels and defined colors

        Parameters
        -----------
        data : pandas.Series, pandas.DataFrame, numpy.ndarray
            Data to be plotted
        style : str
            Boxplot style. Possible:
            * default : Standard boxplot
            * narrow : Boxplot for large amount of data
        by : pandas.Series
            Categories used to split data
        showtext : bool
            Display summary statistics values
        centertext : bool
            Center the text within the boxplot instead of on the side
        linewidth : int
            Plot lines width in points
        width_from_count : bool
            Use counts to define the boxplot width
        digitnumber : int
            Number of digits in number format
        box_coverage : float
            Coverage defining the percentile used to compute box extent.
            Example: coverage = 50% -> 25%/75% whiskers
        whiskers_coverage : float
            Coverage defining the percentile used to compute whiskers extent.
            Example: coverage = 90% -> 5%/95% whiskers
        '''

        # Check input data
        if not by is None:
            by = pd.Series(by)
            if by.name is None:
                by.name = 'by'

            if len(by.unique()) == 1:
                raise BoxplotError('by has 1 category only')

            try:
                data = pd.Series(data).astype(np.float64)
            except Exception as err:
                raise BoxplotError('Failed to convert data to float'+\
                        ' series: {0}'.format(str(err)))

        else:
            try:
                data = pd.DataFrame(data).astype(np.float64)
            except Exception as err:
                raise BoxplotError('Failed to convert data to float'+\
                        ' dataframe:' +\
                        ' {0}'.format(str(err)))

        self._ax = None

        self._data = data

        self._by = by

        self.elements = None

        if box_coverage < 40.:
            raise BoxplotError('Box coverage cannot be below 40. '+\
                'Got {0}.'.format(box_coverage))
        self.box_coverage = box_coverage

        if whiskers_coverage <= box_coverage:
            raise BoxplotError('Whiskers coverage cannot be below box'+\
                ' coverage.'+\
                ' Got {0} for box cov and {1} for whisk cov.'.format(\
                    box_coverage, whiskers_coverage))
        self.whiskers_coverage = whiskers_coverage

        self._width_from_count = width_from_count

        # Configure box plot formats depending on style
        if style == 'default':
            self.median = BoxplotItem(linecolor=COLORS[3], \
                            fontcolor=COLORS[3], fontsize=9, \
                            marker='none',\
                            linewidth=linewidth, \
                            showtext=showtext)

            self.whiskers = BoxplotItem(linecolor=COLORS[0], \
                                facecolor=COLORS[0], \
                                linewidth=linewidth, \
                                width=0.)

            self.caps = BoxplotItem(linecolor=COLORS[0], \
                            linewidth=linewidth, \
                            width=0.3, fontcolor=COLORS[0])

            self.box = BoxplotItem(linecolor=COLORS[0], \
                            width=0.7, fontcolor=COLORS[0], \
                            textformat = '%0.{0}f'.format(digitnumber), \
                            fontsize=8, \
                            linewidth=linewidth, \
                            showtext=showtext)


        elif style == 'narrow':
            self.median = BoxplotItem(marker='o', \
                            markeredgecolor=COLORS[0], \
                            markerfacecolor='w', \
                            showline=False, \
                            showtext=False)

            self.whiskers = BoxplotItem(linecolor='none', \
                                facecolor=COLORS[0], \
                                alpha=0.5, \
                                width=0.3, linewidth=0)

            self.caps = BoxplotItem(linecolor='none', width=0.)

            self.box = BoxplotItem(linecolor='none', \
                            facecolor=COLORS[0], \
                            width=0.6, showtext=False)

        else:
            raise BoxplotError('Expecting style in [default/narrow],'+\
                        ' got {0}'.format(style))

        # Set text format
        for obj in [self.median, self.whiskers, self.box, self.caps]:
            obj.textformat = '%0.{0}f'.format(digitnumber)

        self.centertext = centertext
        if centertext:
            for obj in [self.median, self.box, self.whiskers]:
                obj.ha = 'center'
                obj.va = 'bottom'
            self.box.ha = 'left'

        # Items not affected by style
        self.count = BoxplotItem(fontsize=7, \
                        fontcolor='grey', textformat='%d')

        self.minmax = BoxplotItem(markerfacecolor=COLORS[0], \
                            marker='none', \
                            showline=False)

        # Compute boxplot stats
        self._compute()


    def _compute(self):
        ''' Compute boxplot stats '''

        data = self._data
        by = self._by
        bhc = self.box_coverage
        whc = self.whiskers_coverage

        if by is None:
            self._stats = data.apply(boxplot_stats, args=(bhc, whc, ))
        else:
            stats = data.groupby(by).apply(boxplot_stats, bhc, whc)

            # Reformat to make a 2d dataframe
            stats = stats.reset_index()
            self._stats = pd.pivot_table(stats, \
                index='level_1', columns=by.name, values=stats.columns[-1])


    @property
    def stats(self):
        ''' Returns the boxplot stats '''
        return self._stats


    @property
    def ax(self):
        ''' Returns the boxplot axe '''
        return self._ax


    def draw(self, ax=None, logscale=False, xoffset=0.):
        ''' Draw the boxplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the boxplot on
        logscale : bool
            Use y axis log scale or not
        xoffset : float
            Add an offset to x axis
        '''

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        ax = self._ax
        stats = self._stats
        ncols = stats.shape[1]

        bqq1, bqq2 = compute_percentiles(self.box_coverage)
        bqq1txt = '{0:0.1f}%'.format(bqq1)
        bqq2txt = '{0:0.1f}%'.format(bqq2)

        wqq1, wqq2 = compute_percentiles(self.whiskers_coverage)
        wqq1txt = '{0:0.1f}%'.format(wqq1)
        wqq2txt = '{0:0.1f}%'.format(wqq2)

        # Boxplot widths
        if self._width_from_count:
            cnt = stats.loc['count', :].values
            ratio = cnt/cnt.max()
            boxwidths = ratio*self.box.width
            capswidths = ratio*self.caps.width
            whiskerswidths = ratio*self.whiskers.width
        else:
            boxwidths = np.ones(ncols)*self.box.width
            capswidths = np.ones(ncols)*self.caps.width
            whiskerswidths = np.ones(ncols)*self.whiskers.width

        # Loop through stats
        self.elements = {}
        for i, colname in enumerate(stats.columns):
            # Box Widths
            bw = boxwidths[i]

            # Draw median
            x = [i-bw/2+xoffset, i+bw/2+xoffset]
            med = stats.loc['50.0%', colname]
            y = [med] * 2
            valid_med = np.all(~np.isnan(med))

            item = self.median
            element = {}

            if item.showline and valid_med:
                ax.plot(x, y, lw=item.linewidth, \
                    color=item.linecolor, \
                    alpha=item.alpha)
                element['median-line'] = ax.get_lines()[-1]

            if item.marker != 'none':
                ax.plot(i+xoffset, med, marker=item.marker, \
                    markeredgecolor=item.markeredgecolor, \
                    markerfacecolor=item.markerfacecolor, \
                    markersize=item.markersize, \
                    alpha=item.alpha)
                element['median-marker'] = ax.get_lines()[-1]

            if item.showtext and valid_med:
                formatter = item.textformat
                xshift = 0
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = bw/2

                medtext = formatter % med
                element['median-text'] = ax.text(i+xshift+xoffset, \
                        med, medtext, \
                        fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=item.va, ha=item.ha, \
                        alpha=item.alpha)

            # Skip missing data
            q1 = stats.loc[bqq1txt, colname]
            q2 = stats.loc[bqq2txt, colname]

            valid_q1 = np.all(~np.isnan(q1))
            valid_q2 = np.all(~np.isnan(q2))
            if not valid_q1 or not valid_q2:
                continue

            # Draw box
            item = self.box
            if item.facecolor !='none' or item.showline:
                # Remove rounding to avoid weird spikes
                # when there are too many columns
                boxstyle = item.boxstyle
                if ncols>8 or logscale:
                    boxstyle = re.sub('rounding_size=[^,]+', \
                                'rounding_size=0.', boxstyle)

                bbox = FancyBboxPatch([i-bw/2+xoffset, q1], bw, q2-q1, \
                    boxstyle=boxstyle, \
                    facecolor=item.facecolor, \
                    linewidth=item.linewidth, \
                    edgecolor=item.linecolor,
                    alpha=item.alpha)

                ax.add_patch(bbox)
                element['box'] = ax.patches[-1]

            # Whisker width
            ww = whiskerswidths[i]

            # Draw whiskers
            item = self.whiskers
            if item.showline:
                for cc in [[wqq1txt, bqq1txt], [wqq2txt, bqq2txt]]:
                    # Get y data
                    q1 = stats.loc[cc[0], colname]
                    q2 = stats.loc[cc[1], colname]

                    if ww < EPS:
                        # Draw line
                        x = [i+xoffset]*2
                        y = [q1, q2]
                        ax.plot(x, y, lw=item.linewidth,
                            color=item.linecolor, \
                            alpha=item.alpha)
                        element['bottom-whiskers'] = ax.get_lines()[-2]
                        element['top-whiskers'] = ax.get_lines()[-1]
                    else:
                        # Draw box
                        bbox = FancyBboxPatch([i-ww/2+xoffset, q1], \
                            ww, q2-q1, \
                            boxstyle=item.boxstyle, \
                            facecolor=item.facecolor, \
                            linewidth=item.linewidth, \
                            edgecolor=item.linecolor,
                            alpha=item.alpha)

                        ax.add_patch(bbox)
                        element['whiskers'] = ax.patches[-1]

            # Cap width
            cw = capswidths[i]

            # Draw caps
            item = self.caps
            if item.showline and cw>0:
                for qq in [wqq1txt, wqq2txt]:
                    q1 = stats.loc[qq, colname]
                    x = [i-cw/5+xoffset, i+cw/5+xoffset]
                    y = [q1]*2
                    ax.plot(x, y, lw=item.linewidth,
                        color=item.linecolor, \
                        alpha=item.alpha)

                    element['bottom-cap'] = ax.get_lines()[-2]
                    element['top-cap'] = ax.get_lines()[-1]

            # Box (quartile) values
            item = self.box
            if item.showtext:
                formatter = item.textformat
                if item.ha == 'left':
                    formatter = ' '+formatter
                    xshift = bw/2
                elif item.ha == 'center':
                    xshift = 0

                values = [stats.loc[qq, colname] \
                                    for qq in [bqq1txt, bqq2txt]]

                for ivalue, value in enumerate(values):
                    valuetext = formatter % value

                    # Slight realignment of label for centered text option
                    va, ha = item.va, item.ha
                    if self.centertext:
                        xshift = 0
                        va = 'top' if ivalue == 0 else 'bottom'

                    # Draw text
                    element['box-text'] = ax.text(i+xshift+xoffset, \
                        value, valuetext, \
                        fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=va, ha=ha, \
                        alpha=item.alpha)

            # Draw min / max
            item = self.minmax
            if item.marker != 'none':
                x = [i+xoffset]*2
                y = stats.loc[['min', 'max'], colname]
                ax.plot(x, y, item.marker, \
                    mfc=item.markerfacecolor, \
                    mec=item.markeredgecolor, \
                    markersize=item.markersize, \
                    alpha=item.alpha)

                element['minmax'] = ax.get_lines()[-1]

            # Store element
            self.elements[colname] = element

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
            ylim0 = min(ylim[0], stats.loc[wqq1txt, :].min()-dy*0.02)
            ylim1 = max(ylim[1], stats.loc[wqq2txt, :].max()+dy*0.02)
        else:
            dy = math.log(ylim[1])-math.log(ylim[0])
            ylim0 = ylim[0]
            miniq = stats.loc[wqq1txt, :].min()
            if miniq>0:
                ylim0 = min(ylim[0], math.exp(math.log(miniq)-dy*0.02))

            ylim1 = ylim[1]
            maxiq = stats.loc[wqq1txt, :].max()
            if maxiq>0:
                ylim1 = max(ylim[1], math.exp(math.log(maxiq)+dy*0.02))

        ax.set_ylim((ylim0, ylim1))


    def show_count(self, ypos=0.025):
        ''' Show the counts '''

        ax = self._ax
        if ax is None:
            raise BoxplotError('Boxplot ax is not initialised.'+\
                ' Run "draw" method first.')

        if self.elements is None:
            raise BoxplotError('No elements in the boxplot. Run draw')

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
                txt = ax.text(i, y, cnt, fontsize=item.fontsize, \
                        color=item.fontcolor, \
                        va=va, ha='center')

                if cn in self.elements:
                    self.elements[cn]['count-text'] = txt

        else:
            raise BoxplotError('showtext property for count set to False')

