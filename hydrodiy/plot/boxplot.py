from datetime import datetime
import math
import numpy as np
import pandas as pd

from hydrodiy.stat import sutils
from hydrodiy.plot import putils

COLORS = putils.tableau_colors

PROPS_NAMES = ['median', 'whisker', 'box', 'count', 'minmax']

PROPS_VALUES = ['linestyle', 'linewidth', 'linecolor', 'va', 'ha', 'fontcolor', \
    'textformat', 'fontsize', 'marker', 'markercolor']


def perct(data):
    idx = pd.notnull(data)
    prc = sutils.percentiles(data[idx], [10, 25, 50, 75, 90])
    prc['count'] = np.sum(idx)
    prc['max'] = data[idx].max()
    prc['min'] = data[idx].min()
    return prc


class Boxplot(object):

    def __init__(self, data, ax=None, by=None, default_width=0.7, \
                width_from_number=False, show_values=True, \
                show_minmax=False):
        ''' Draw boxplots with labels and defined colors '''

        if not by is None:
            if not isinstance(by, pd.Series):
                raise ValueError('by is not a Series')
            if not isinstance(data, pd.Series):
                raise ValueError('data is not a Series')
        self._data = data
        self._by = by

        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        self._default_width = default_width

        self._width_from_number = width_from_number

        self._show_values = show_values
        self._show_minmax = show_minmax

        self._props = {
            'median':{'linestyle':'-', 'linewidth':3, 'linecolor':COLORS[3], \
                'va':'center', 'ha':'left', 'fontcolor':COLORS[3], \
                'textformat':'%0.1f', 'fontsize':9}, \
            'whisker':{'linestyle':'-', 'linewidth':2, 'linecolor':COLORS[0]}, \
            'minmax':{'marker':'o', 'markercolor':COLORS[0]}, \
            'box':{'linestyle':'-', 'linewidth':3, 'linecolor':COLORS[0], \
                'va':'center', 'ha':'left', 'fontcolor':COLORS[0], \
                'textformat':'%0.1f', 'fontsize':8}, \
            'count':{'fontcolor':'grey', 'textformat':'%d', \
                'fontsize':7},
        }

        self._compute()


    def __setitem__(self, key, value):
        ''' Set boxplot properties '''
        if not key in PROPS_NAMES:
            raise ValueError('Cannot set properties for '+key)

        for k in value:
            if not k in PROPS_VALUES:
                raise ValueError('Cannot set values '+k)
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
        show_values = self._show_values
        show_minmax = self._show_minmax

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

            ax.plot(x, y, lw=props['linewidth'],
                color=props['linecolor'])

            if show_values:
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

            ax.plot(x, y, lw=props['linewidth'],
                color=props['linecolor'])

            # Draw whiskers and caps
            props = self._props['whisker']

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
            if show_values:
                formatter = props['textformat']
                if props['ha'] == 'left':
                    formatter = ' '+formatter
                for value in stats.loc[['25.0%', '75.0%'], cn]:
                    valuetext = formatter % value
                    ax.text(i+w/2, value, valuetext, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va=props['va'], ha=props['ha'])

            # min / max
            if show_minmax:
                x = [i]*2
                y = stats.loc[['min', 'max'], cn]
                props = self._props['minmax']
                ax.plot(x, y, props['marker'], color=props['markercolor'])

        # X tick labels
        ax.set_xticks(range(ncols))
        ax.set_xticklabels(stats.columns)

        ax.set_xlim((-default_width, ncols-1+default_width))

        if not self._by is None:
            ax.set_xlabel(self._by.name)

        # Y axis log scale
        if logscale:
            ax.set_yscale('log', nonposy='clip')

        # Display count
        if show_values:
            props = self._props['count']
            formatter = '('+props['textformat']+')'

            ylim = ax.get_ylim()
            y = ylim[0] + 0.01*(ylim[1]-ylim[0])
            for i, cn in enumerate(stats.columns):
                cnt = formatter % stats.loc['count', cn]
                ax.text(i, y, cnt, fontsize=props['fontsize'], \
                        color=props['fontcolor'], \
                        va='bottom', ha='center')

        return
