import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties

from hyplot import putils

from calendar import month_abbr

# Axis labels
months = [m[0] for m in month_abbr[1:]] 
seasons = [''.join([months[j] for j in [i, (i+1)%12, (i+2)%12]]) for i in range(len(months))]

# Wafari colors
rgba00 = colors.hex2color("#FF9933")
rgba05 = colors.hex2color("#CCCCCC")
rgba10 = colors.hex2color("#0046AD")
cdict = {
    "red": (
        (0.0, rgba00[0], rgba00[0]),
        (0.4, rgba05[0], rgba05[0]),
        (1.0, rgba10[0], rgba10[0]),
        ),
    "green": (
        (0.0, rgba00[1], rgba00[1]),
        (0.4, rgba05[1], rgba05[1]),
        (1.0, rgba10[1], rgba10[1]),
        ),
    "blue": (
        (0.0, rgba00[2], rgba00[2]),
        (0.4, rgba05[2], rgba05[2]),
        (1.0, rgba10[2], rgba10[2]),
        ),
    }
cmap_wafari = colors.LinearSegmentedColormap('wafari', cdict, 256)

# Wafari fonts
font_super_title = FontProperties(family="sans-serif", size="large", weight="bold")
font_title = FontProperties(family='sans-serif', size="medium")
font_legend = FontProperties(family='sans-serif', size="small")
font_axis = FontProperties(family='sans-serif', size="medium")
font_tick = FontProperties(family='sans-serif', size="x-small")



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


def summary(scores, ax=None, seasonal=True, title=None, ylim=(-5, 70),
            descriptions=None, cmap=None):
    ''' Skill score summary plot

            :parameter pandas.DataFrame scores: Skill score values for n sites and 12 months ([n sites x 12 months] dataframe)
            :parameter string title: Plot title
            :parameter bool seasonal: Seasonal data or monthly?
            :parameter tuple ylim: Min max limits for Y axis
            :parameter list descriptions: Site names
            :parameter matplotlib.colors.LinerSegmentedColormap cmap: Color map 
    '''
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = 'Skill score summary'

    if cmap is None:
        cmap = cmap_wafari

    if descriptions is None:
        descriptions = range(1, scores.shape[0]+1)

    if scores.shape[1] != 12:
        return ValueError('score matrix does not have 12 columns (=%d)' % scores.shape[1])

    if not descriptions is None:
        if len(descriptions) != scores.shape[0]:
            return ValueError('description vector does not have %d elements' % scores.shape[0])

    # Plot data
    x = np.arange(13)
    pc = ax.pcolor(scores, cmap=cmap, vmin=ylim[0], vmax=ylim[1], edgecolor="white")

    # Decorations
    ax.set_xticks(x+0.5)
    if seasonal:
        ax.set_xticklabels(seasons, fontproperties=font_tick)
    else:
        ax.set_xticklabels(months, fontproperties=font_tick)

    ax.set_yticks(np.arange(0.0,len(descriptions))+0.5)
    ax.set_yticklabels(descriptions, fontproperties=font_tick)

    ax.set_xlabel('Forecast period')
    ax.set_title(title, fontproperties=font_title)
    ax.set_xlim((0, 12))

    plt.setp(ax.xaxis.get_ticklines(), visible=False)
    plt.setp(ax.yaxis.get_ticklines(), visible=False)
    ax.set_frame_on(False)

    return pc
