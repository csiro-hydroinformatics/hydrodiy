from datetime import datetime
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

COLORS = putils.COLORS10

PROPS_NAMES = ['median', 'whisker', 'box', 'count', 'minmax']

PROPS_VALUES = ['linestyle', 'linewidth', 'linecolor', 'va', 'ha', 'fontcolor', \
    'textformat', 'fontsize', 'marker', 'markercolor', 'showline', 'showtext']

def whiskers_percentiles(coverage):
    ''' Compute whiskers percentiles from coverage '''
    qq1 = float(100-coverage)/2
    qq2 = 100.-qq1
    return qq1, qq2


def boxplot_stats(data, coverage):
    ''' Compute boxplot stats '''

    idx = pd.notnull(data)
    qq1, qq2 = whiskers_percentiles(coverage)

    if idx.shape[0]>3:
        qq = [qq1, 25, 50, 75, qq2]
        prc = pd.Series(np.percentile(data[idx], qq), \
                        index=['{0:0.1f}%'.format(qqq) for qqq in qq])
        prc['count'] = np.sum(idx)
        prc['mean'] = data[idx].mean()
        prc['max'] = data[idx].max()
        prc['min'] = data[idx].min()
    else:
        prc = pd.Series({'count': idx.shape[0]})
        pnames = ['{0:0.1f}%'.format(qq1), '25.0%', \
                '50.0%', '75.0%', '{0:0.1f}%'.format(qq2), \
                'min', 'max', 'mean']
        for pn in pnames:
            prc[pn] = np.nan

    return prc


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
        self._whiskers_coverage = whiskers_coverage

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        self._default_width = default_width

        self._width_from_count = width_from_count

        self._props = {
            'median':{'linestyle':'-', 'linewidth':3, 'linecolor':COLORS[3], \
                'va':'center', 'ha':'left', 'fontcolor':COLORS[3], \
                'textformat':'%0.1f', 'fontsize':9, 'showline':True, \
                'showtext':True}, \
            'whisker':{'linestyle':'-', 'linewidth':2, 'linecolor':COLORS[0], \
                            'showline':True}, \
            'minmax':{'marker':'o', 'markercolor':COLORS[0], 'showline':False}, \
            'box':{'linestyle':'-', 'linewidth':3, 'linecolor':COLORS[0], \
                'va':'center', 'ha':'left', 'fontcolor':COLORS[0], \
                'textformat':'%0.1f', 'fontsize':8, 'showline':True, \
                'showtext':True}, \
            'count':{'fontcolor':'grey', 'textformat':'%d', \
                'fontsize':7, 'showtext':True},
        }

        self.logscale = False
        self._compute()


    def __setitem__(self, key, value):
        ''' Set boxplot properties '''
        if not key in PROPS_NAMES:
            raise ValueError('Cannot set property '+key)

        for k in value:
            if not k in PROPS_VALUES:
                raise ValueError('Cannot set value '+k)
            self._props[key][k] = value[k]


    def _compute(self):
        ''' Compute boxplot stats '''

        data = self._data
        by = self._by
        whc = self._whiskers_coverage

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


    def set_all(self, propname, value):
        ''' Set the a particular property (e.g. textformat) for all items '''

        # Check how many properties are actually set
        is_set = 0

        for key, item in self._props.iteritems():
            if propname in item:
                item[propname] = value
                is_set += 1

        # Check property exists
        if is_set == 0:
            raise ValueError('Property {0} was never set'.format(propname))


    def draw(self, logscale=False):
        ''' Draw the boxplot '''

        ax = self._ax
        stats = self._stats
        default_width = self._default_width
        ncols = stats.shape[1]
        self.logscale = logscale

        qq1, qq2 = whiskers_percentiles(self._whiskers_coverage)
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
            props = self._props['median']

            valid_med = np.all(~np.isnan(med))

            if props['showline'] and valid_med:
                ax.plot(x, y, lw=props['linewidth'],
                    color=props['linecolor'])

            if props['showtext'] and valid_med:
                formatter = props['textformat']
                if props['ha'] == 'left':
                    formatter = ' '+formatter
                    xshift = w/2
                elif props['ha'] == 'center':
                    xshift = 0

                medtext = formatter % med
                ax.text(i+xshift, med, medtext, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va=props['va'], ha=props['ha'])

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
            props = self._props['box']

            if props['showline']:
                ax.plot(x, y, lw=props['linewidth'],
                    color=props['linecolor'])

            # Draw whiskers and caps
            props = self._props['whisker']

            if props['showline']:
                for cc in [[qq1txt, '25.0%'], [qq2txt, '75.0%']]:
                    q1 = stats.loc[cc[0], cn]
                    q2 = stats.loc[cc[1], cn]

                    x = [i]*2
                    y = [q1, q2]

                    ax.plot(x, y, lw=props['linewidth'],
                        color=props['linecolor'])

                    x = [i-w/5, i+w/5]
                    y = [q1]*2
                    ax.plot(x, y, lw=props['linewidth'],
                        color=props['linecolor'])


            # quartile values
            props = self._props['box']
            if props['showtext']:

                formatter = props['textformat']
                if props['ha'] == 'left':
                    formatter = ' '+formatter
                    xshift = w/2
                elif props['ha'] == 'center':
                    xshift = 0

                for value in [stats.loc[qq, cn] for qq in ['25.0%', '75.0%']]:
                    valuetext = formatter % value
                    ax.text(i+xshift, value, valuetext, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va=props['va'], ha=props['ha'])

            # min / max
            props = self._props['minmax']
            if props['showline']:
                x = [i]*2
                y = stats.loc[['min', 'max'], cn]
                ax.plot(x, y, props['marker'], color=props['markercolor'])

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

    def count(self):
        ''' Show the counts '''

        ax = self._ax
        stats = self._stats
        default_width = self._default_width
        props = self._props['count']

        if props['showtext']:
            formatter = '('+props['textformat']+')'

            ylim = ax.get_ylim()

            if not self.logscale:
                y = ylim[0] + 0.01*(ylim[1]-ylim[0])
            else:
                y = math.exp(math.log(ylim[0]) + \
                            0.01*(math.log(ylim[1])-math.log(ylim[0])))

            for i, cn in enumerate(stats.columns):
                cnt = formatter % stats.loc['count', cn]
                ax.text(i, y, cnt, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va='bottom', ha='center')
        else:
            raise ValueError('showtext property for count set to False')

