from datetime import datetime
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

COLORS = putils.tableau_colors

PROPS_NAMES = ['median', 'whisker', 'box', 'count', 'minmax']

PROPS_VALUES = ['linestyle', 'linewidth', 'linecolor', 'va', 'ha', 'fontcolor', \
    'textformat', 'fontsize', 'marker', 'markercolor', 'showline', 'showtext']


def perct(data):
    idx = pd.notnull(data)
    prc = sutils.percentiles(data[idx], [10, 25, 50, 75, 90])
    prc['count'] = np.sum(idx)
    prc['max'] = data[idx].max()
    prc['min'] = data[idx].min()
    return prc


class Boxplot(object):

    def __init__(self, data, ax=None, by=None, default_width=0.7, \
                width_from_number=False):
        ''' Draw boxplots with labels and defined colors '''

        if not by is None:
            by = pd.Series(by)
            data = pd.Series(data)
        else:
            data = pd.DataFrame(data)

        self._data = data
        self._by = by

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        self._default_width = default_width

        self._width_from_number = width_from_number

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

        if by is None:
            self._stats = data.apply(perct)
        else:
            stats = data.groupby(by).apply(perct)
            stats = stats.reset_index()
            self._stats = pd.pivot_table(stats, \
                index='level_1', columns=by.name, values=stats.columns[-1])


    def draw(self, logscale=False):
        ''' Draw the boxplot '''

        ax = self._ax
        stats = self._stats
        default_width = self._default_width
        ncols = stats.shape[1]

        # Boxplot widths
        if self._width_from_number:
            cnt = stats['count']
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

            if props['showline']:
                ax.plot(x, y, lw=props['linewidth'],
                    color=props['linecolor'])

            if props['showtext']:
                formatter = props['textformat']
                if props['ha'] == 'left':
                    formatter = ' '+formatter
                medtext = formatter % med
                ax.text(i+w/2, med, medtext, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va=props['va'], ha=props['ha'])

            # Draw boxes
            x = [i-w/2, i+w/2, i+w/2, i-w/2, i-w/2]
            q1 = stats.loc['25.0%', cn]
            q2 = stats.loc['75.0%', cn]
            y =  [q1, q1, q2, q2 ,q1]
            props = self._props['box']

            if props['showline']:
                ax.plot(x, y, lw=props['linewidth'],
                    color=props['linecolor'])

            # Draw whiskers and caps
            props = self._props['whisker']

            if props['showline']:
                for cc in [['10.0%', '25.0%'], ['90.0%', '75.0%']]:
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

                for value in stats.loc[['25.0%', '75.0%'], cn]:
                    valuetext = formatter % value
                    ax.text(i+w/2, value, valuetext, fontsize=props['fontsize'], \
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
        dy = ylim[1]-ylim[0]
        ylim0 = min(ylim[0], stats.loc['10.0%', :].min()-dy*0.02)
        ylim1 = max(ylim[1], stats.loc['90.0%', :].max()+dy*0.02)
        ax.set_ylim((ylim0, ylim1))

        # Display count
        props = self._props['count']
        if props['showtext']:
            formatter = '('+props['textformat']+')'

            ylim = ax.get_ylim()
            y = ylim[0] + 0.01*(ylim[1]-ylim[0])
            for i, cn in enumerate(stats.columns):
                cnt = formatter % stats.loc['count', cn]
                ax.text(i, y, cnt, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va='bottom', ha='center')

        return
