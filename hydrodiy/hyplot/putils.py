import os
import re
from datetime import datetime
import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

from hystat import sutils


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

def footer(fig, author, version=None, type='bureau'):
    ''' Add footer to a figure (code from wafari) '''
    now = datetime.datetime.now()
    label = '%s - Generated: %s' % (author, now.strftime('%H:%M %d/%m/%Y'))

    if not version is None:
        label = label + ' (ver. %s)'%version

    # Add label
    fig.text(0.05, 0.010, label, color='#595959', ha='left', fontsize=9)

    # Add copyright
    if type=='bureau':
        copyright = u'\u00A9' 
        copyright += 'Commonwealth of Australia %s'%now.strftime('%Y')
        copyright += ' Australian Bureau of Meteorology'
        fig.text(0.95, 0.010, copyright, color='#595959', 
                                ha='right', fontsize=9)
    elif type=='jl':
        copyright = u'\u00A9' 
        copyright += 'Julien Lerat %s'%now.strftime('%Y')
        fig.text(0.95, 0.010, copyright, color='#595959', 
                                ha='right', fontsize=9)


def rectangle_paths(left, right, bottom, top, 
                        rounded_fraction=0.3):
    ''' 
        build a Path object containing multiple 
        rectangles. The rounded option uses rounded
        corners as a fraction min(width, height).
        Should be smaller than 0.5
    '''
    # see http://matplotlib.org/examples/pylab_examples/fancybox_demo.html
    # should be much easier !!!

    # Determines rounded corner size
    wx = np.min(right-left)
    rx = np.max(right)-np.min(left)
    wy = np.min(top-bottom)
    ry = np.max(top)-np.min(bottom)
    ww = min(wx/rx, wy/ry)

    rwidthx = min(0.5, rounded_fraction) * ww * rx 
    rwidthy = min(0.5, rounded_fraction) * ww * ry
   
    # initialise path object
    npt = 13
    nverts = len(left) * npt
    verts = np.zeros((nverts,2))
    codes = np.ones(nverts, int) * Path.LINETO
    
    codes[0::npt] = Path.MOVETO
    codes[(npt-1)::npt] = Path.CLOSEPOLY
   
    # Rounded corners
    for k in [2, 3, 5, 6, 8, 9, 11, 12]:
        codes[k::npt] = Path.CURVE3
   
    # Draw path
    verts[0::npt,0] = left
    verts[0::npt,1] = bottom+rwidthy
    
    verts[1::npt,0] = left
    verts[1::npt,1] = top-rwidthy
    
    verts[2::npt,0] = left
    verts[2::npt,1] = top
    
    verts[3::npt,0] = left+rwidthx
    verts[3::npt,1] = top
    
    verts[4::npt,0] = right-rwidthx
    verts[4::npt,1] = top
    
    verts[5::npt,0] = right
    verts[5::npt,1] = top
    
    verts[6::npt,0] = right
    verts[6::npt,1] = top-rwidthy
    
    verts[7::npt,0] = right
    verts[7::npt,1] = bottom+rwidthy
    
    verts[8::npt,0] = right
    verts[8::npt,1] = bottom
    
    verts[9::npt,0] = right-rwidthx
    verts[9::npt,1] = bottom
    
    verts[10::npt,0] = left+rwidthx
    verts[10::npt,1] = bottom
    
    verts[11::npt,0] = left
    verts[11::npt,1] = bottom
    
    verts[12::npt,0] = left
    verts[12::npt,1] = bottom+rwidthy

    return Path(verts, codes)

def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, 
    except that it also plots a proxy artist 
    (specifically, a rectangle of 0 size) so that it can be added 
    it appears on a legend.
    See http://goo.gl/tGLbji 
    """
    ax = ax if ax is not None else plt.gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = plt.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)

    return p

def plot_ensembles(ensembles, ax, 
        percentiles=[10., 25.],
        xx=None, line_width=2, alpha=0.9):
    ''' 
        Plot percentiles of ensembles 
        
        :param numpy.array ensembles : Data frame containing ensembles in columns
        :param matplotlib.axes ax: Axe to draw ensembles on
        :param list percentiles: List of low percentiles to compute 
                        ensemble stats. Quantiles should be <50. 
                        the 50 and other symetric percentiles are added
                        automatically 
                        (e.g. [10, 20] -> [10, 20, 50, 80, 90])
        :param numpy.array xx: Abscissae to use for plotting 
        :param float line_width: Width of line drawn
        :param float alpha: Transparency
    '''
   
    # compute percentiles
    qt = percentiles+[50]+[100-q for q in percentiles]
    args = (qt,)
    ens = pd.DataFrame(ensembles)
    ens_qq = ens.apply(sutils.percentiles, 
            args=args, axis=1)

    # Ensemble colors
    ncols = 2+len(percentiles)
    cols = get_colors(ncols, 'Blues')[1:]

    # Rearrange columns 
    ens_qq = ens_qq[np.sort(ens_qq.columns.values)]

    # dimensions
    nval = ens_qq.shape[0]
    nens = ens_qq.shape[1]

    # Abscissae
    if xx is None:
        u = ens_qq.index.values
        x = np.hstack([u, u[::-1], u[0]])
    else:
        x = np.hstack([xx, xx[::-1], xx[0]])

    # plots ensemble range
    columns = ens_qq.columns
    for i in range(len(cols)-1):
        v1 = ens_qq[columns[i]]
        v2 = ens_qq[columns[nens-i-1]]
        lab = '%s to %s'%(columns[i], columns[nens-i-1])
        lab = re.sub('_', ' ', lab)
        fill_between(ens_qq.index.values ,v1.values, 
                v2.values, ax=ax, 
                color=cols[i], lw=0.1, alpha=0.8, 
                label=lab)
    
    # plot median
    ax.plot(ens_qq.index,
            ens_qq[columns[len(cols)-1]], 
            lw=line_width,
            color=cols[len(cols)-1], 
            label=re.sub('_', ' ',columns[len(cols)-1]))

    return ens_qq

def label(ax, label, fontsize=16):
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
           fontsize=fontsize, fontweight='bold', va='top')


