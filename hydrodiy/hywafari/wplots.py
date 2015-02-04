import numpy as np

import matplotlib.pyplot as plt

from hyplot import putils

from calendar import month_abbr
months = [m[0] for m in month_abbr[1:]] 
seasons = [''.join([months[j] for j in [i, (i+1)%12, (i+2)%12]]) for i in range(len(months))]


def skillscores(scores, ax=None, seasonal=True, title=None, ylim=(-10, 70)):
        ''' Skill score plot

            :parameter pandas.DataFrame scores: Skill score values for 12 months ([12 x n scores] dataframe)
            :parameter string title: Plot title
            :parameter bool seasonal: Seasonal data or monthly?
            :parameter tuple ylim: Min max limits for Y axis
        '''

        if ax is None:
            ax = plt.gca()

        if title is None:
            title = 'Skill score summary'

        if scores.shape[0] != 12:
            return ValueError('score matrix does not have 12 lines (=%d)' % scores.shape[0])
   
        # Plot data
        width = 0.8
        xx = np.arange(12)
        scorenames = scores.columns
        ns = len(scorenames)
        colors = putils.get_colors(ns+2, 'Blues')[1:]

        # Decorations
        ax.set_xticks(xx+width/2)
        if seasonal:
            ax.set_xticklabels(seasons)
        else:
            ax.set_xticklabels(months)

        ax.grid(lw=0.25)
        ax.set_xlabel('Forecast period')
        ax.set_ylabel('Skill score (%)')
        ax.set_title(title) 

        # Plot
        for i in range(ns):
            ax.bar(xx+width*float(i)/ns, scores[scorenames[i]], width/ns, 
                color=colors[i], label=scorenames[i])

        ax.legend(loc=1, fancybox=True, framealpha=0.9)

        # Limits
        ax.set_ylim(ylim)
        ax.set_xlim((-width/2, 12+width/4))


