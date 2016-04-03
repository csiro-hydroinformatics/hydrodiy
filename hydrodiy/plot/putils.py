import os
import re
from cycler import cycler
from datetime import datetime
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.path import Path

from matplotlib import colors
from matplotlib.colors import hex2color
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd

# Some useful colors
wafari_tercile_colors = ['#FF9933', '#64A0C8', '#005BBB']

bureau_background_color = '#002745'

tableau_colors = [colors.rgb2hex([float(coo)/255 for coo in co]) for co in [ \
            (31, 119, 180), (255, 127, 14), (44, 160, 44), \
            (214, 39, 40), (148, 103, 189), (140, 86, 75), \
            (227, 119, 194), (127, 127, 127), (188, 189, 34), \
            (23, 190, 207)
        ] ]

def set_color_cycle(ax, cycle_colors=None):
    if cycle_colors is None:
        cycle_colors = tableau_colors
    ax.set_prop_cycle(cycler('color', cycle_colors))


def get_colors(ncols=10, palette='Paired'):
    ''' generates a set of colors '''
    cmap = cm.get_cmap(palette, ncols)
    return [cmap(i) for i in range(cmap.N)]


# Fucntion to generate html code
def img2html(title, image_data, root_http=None, filename=None):
    '''
        Generate html code gathering various image in a big table

        :param str title : page title
        :param pandas.DataFrame image_data : Data used to access image files
            image_data.columns = Column header
            image_data.index = Row header
            image_data['rowlabel'] = Text at the beginning of each row
        :param str root_http : HTTP adress of root directory containing images
        :param string filename: Output file name to write

    '''

    has_rowlabel = 'rowlabel' in image_data.columns

    now = datetime.datetime.now().date().isoformat()
    html = ['<html>','<head>','<title> %s </title>'%title,
        '<b><p>Root HTTP : %s</p>'%root_http,
        '<p>Generated on : %s</p>'%now,
        '<p>Author : %s</p>'%os.getlogin(),
        '</head>\n','<body>\n<hr>\n']
    html.append('<table>\n') # border=1 style="width:300px">\n')
    html.append('<tr><td></td></tr>\n')

    # urls data
    colurl = [cn for cn in image_data.columns
            if cn!='rowlabel']

    # column headers
    html.append('<tr>')
    html.append('<td align=\'center\'>index</td>\n')
    if has_rowlabel:
        html.append('<td align=\'center\'>Label</td>\n')
    for cn in colurl:
        html.append('<td align=\'center\'>%s</td>\n'%cn)
    html.append('</tr>')
    html.append('<tr><td></td></tr>\n')


    for idx, row in image_data.iterrows():
        html.append('<tr>')
        html.append('<td>%s</td>\n'%idx)
        if has_rowlabel:
            html.append('<td>%s</td>\n'%row['rowlabel'])

        for cn in colurl:
            if not root_http is None:
                html.append('<td><img src="%s/%s"/></td>\n'%(root_http,
                                                                row[cn]))
            else:
                html.append('<td><img src="%s"/></td>\n'%row[cn])

        html.append('</tr>\n')

    html.append('</table>')
    html.append('</body>')
    html.append('</html>')

    if not filename is None:
        with open(filename, 'w') as fout:
            fout.writelines(html)

    return html


def footer(fig, author=None, version=None):
    ''' Add footer to a figure (code from wafari) '''
    now = datetime.datetime.now()

    if not author is None:
        label = '%s - Generated: %s' % (author,
                now.strftime('%H:%M %d/%m/%Y'))
    else:
        label = 'Generated: %s' % now.strftime('%H:%M %d/%m/%Y')

    if not version is None:
        label = label + ' (ver. %s)'%version

    # Add label
    fig.text(0.05, 0.010, label, color='#595959', ha='left', fontsize=9)


def col2cmap(colors):
    ''' Define a linear color map from a set of colors

    Parameters
    -----------
    colors : dict
        A set of colors indexed by a float in [0, 1]. The index
        provides the location in the color map. Example:
        colors = {'0.':'#3399FF', '0.1':'#33FFFF', '1.0':'#33FF99'}

    Returns
    -----------
    cmap : matplotlib.colormap
        Colormap

    Example
    -----------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from hyplot import putils
    >>> colors = {0.:'#3399FF', 0.1:'#33FFFF', 1.0:'#33FF99'}
    >>> cmap = putils.col2cmap(colors)
    >>> nval = 500
    >>> x = np.random.normal(size=nval)
    >>> y = np.random.normal(size=nval)
    >>> z = np.random.uniform(0, 1, size=nval)
    >>> plt.scatter(x, y, c=z, cmap=cmap)

    '''
    keys = np.sort(colors.keys()).astype(float)

    if keys[0] < 0.:
        raise ValueError('lowest key(%f) is lower than 0' % keys[0])

    if keys[-1] > 1.:
        raise ValueError('lowest key(%f) is lower than 0' % keys[-1])

    cdict = {
            'red': [],
            'green': [],
            'blue': []
        }

    for k in keys:
        col = hex2color(colors[k])

        cdict['red'].append((k, col[0], col[0]))
        cdict['green'].append((k, col[1], col[1]))
        cdict['blue'].append((k, col[2], col[2]))

    return LinearSegmentedColormap('mycmap', cdict, 256)


def line(ax, a, b, *args, **kwargs):
    ''' Plot a line y = a + bx
        If b=np.inf, draw a vertical line x=a

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw the line on
    a : float
        Intercept
    b : float
        Slope

    Returns
    -----------
    line : matplotlib.lines.Line2D
        Line drawn

    Example
    -----------
    >>> import matplotlib.pyplot as plt
    >>> from hyplot import putils
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 10], [0, 10], 'o')
    >>> putils.line(0, 1, ax, '--')
    >>> putils.line(0, np.inf, ax, '-', color='red')

    '''

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    a = float(a)
    b = float(b)

    xx = np.array([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])])

    if np.isinf(b):
        line = ax.plot(np.array([a]*2), xx, *args, **kwargs)
    else:
        yy = a + b * xx
        line = ax.plot(xx, yy, *args, **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return line


def equation(tex, filename, \
    textcolor='white', \
    transparent=True, \
    dpi = 200, \
    width = 1000, \
    height = 300):
    ''' Print latex equation into file

    Parameters
    -----------
    tex : str
        Latex equation code
    filename : str
        Filename to print in
    textcolor : str
        Text color
    transparent : bool
        Use transparent background or not
    dpi : int
        Figure resolution
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    fontsize : int
        Font size in points
    Example
    -----------
    >>> from hyplot import putils
    >>> tex = r'\begin{equation} s = \sum_{i=0}^{\infty} \frac{1}{i^2}
    >>> fp = '~/equation.png'
    >>> putils.equation(tex, fp)

    '''
    usetex = mpl.rcParams['text.usetex']
    mpl.rc('text', usetex=True)

    plt.close('all')

    fig, ax = plt.subplots()

    ax.text(0, 0.5, tex, color=textcolor, \
        fontsize=32, va='center')
    ax.set_ylim([0, 1.5])

    ax.axis('off')

    fig.set_size_inches(float(width)/dpi, \
                    float(height)/dpi)

    fig.tight_layout()

    fig.savefig(filename, dpi=dpi, \
        transparent=transparent)

    mpl.rc('text', usetex=usetex)


def set_spines(ax, spines, color='black', style='-', visible=True):
    if spines == 'all':
        spines = ['top', 'bottom', 'left', 'right']

    styles = { \
            ':':'dotted', \
            '-':'solid', \
            '-.':'dash_dot', \
            '--':'dashed'\
    }

    for spine in spines:
        ax.spines[spine].set_visible(visible)
        ax.spines[spine].set_color(color)

        s = style
        if style in [':', '-', '-.', '--']:
            s = styles[style]
        ax.spines[spine].set_linestyle(s)



def set_legend(leg, textcolor='black', alpha=1):
    leg.get_frame().set_alpha(alpha)

    for text in leg.get_texts():
        text.set_color(textcolor)

