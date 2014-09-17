
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm

import hystat
import hyplot

class Boxplot:
    def __init__(self, data, varcat=None):

        self.data = data
        self.varcat = varcat
        self.percentiles = [10, 25, 50]

        # Boxplot aspect
        blues = hyplot.utils.get_colors(20, 'Blues')
        reds = hyplot.utils.get_colors(20, 'Reds')

        self._extreme_box = {'width':0.2, 'facecolor':blues[10], 
                'edgecolor':blues[10], 'lw':2, 'alpha':0.8}
        self._middle_box = {'width':0.5, 'facecolor':blues[15], 
                'edgecolor':blues[15], 'lw':2, 'alpha':1.0}

        self._centre_box = {'width':0.7, 'facecolor':reds[10], 
                'edgecolor':reds[15], 'lw':4, 'alpha':1.0}

        # Compute plotting data
        self.toplot = self._compute_plotting_data(self)

    def _compute_plotting_data(self):
        percentiles = sorted(self.percentiles+[100-self.percentiles[:2]])

        if self.varcat is not None:
            grp = self.data.groupby(by=self.varcat[1])[self.varcat[0]]
            numbers = grp.apply(len)
            toplot = grp.apply(hystat.sutils.percentiles, percentiles)
            toplot = toplot.unstack()
        else:
            toplot = self.data.apply(hystat.utils.compute_percentiles, 
                                        args=[percentiles]).T

        return toplot

    def boxplot(self, ax, print_median=False,
                print_number=False):
        ''' Compute and draw box plots 

        :param matplotlib.pyplot.axes ax: axes to draw boxplot on
        :param bool print_median : print median values on plot 
        :param bool print_number : print number of value on plot 
    
    '''
   
    # Set dimensions
    var_names = self.toplot.index
    nv = len(var_names)
    quant_names = self.toplot.columns
    nq = len(quant_names)
    x = np.arange(nv)
    
    # draw the rectangles for each quantile
    # see http://matplotlib.org/users/path_tutorial.html 
    nql = int(nq/2)+1
    qrects = []
    for box in [self._extreme_box, self._middle_box,
                self._centre_box]:

        # coordinates of rectangle
        width = box['width']
        left = x-width/2
        right = x+width/2
        bottom = toplot[quant_names[iq]]
        top = toplot[quant_names[nq-1-iq]]

        # build path object
        quantpath = hyplot.utils.rectangle_paths(left, right, 
                        bottom, top)

        # Draw plot
        qpatch = patches.PathPatch(quantpath, 
                            lw=box['lw'],
                            facecolor=box['facecolor'],
                            edgecolor=box['edgecolor'],
                            alpha=box['alpha'])
            
        ax.add_patch(qpatch)
    
        # fine tuning of plot 
        ax.set_xlim((-0.5, nv-0.5))
        y0 = np.min(toplot.min())
        y1 = np.max(toplot.max())
        ax.set_ylim((y0-(y1-y0)/10, y1+(y1-y0)/10))
       
        # xlabel
        ax.set_xticks(x)
        ax.set_xticklabels(var_names)

        # text
        if print_median:
            medians = toplot[toplot.columns[nql-1]].values
            delta = (y1-y0)/50
            for i in range(len(medians)):
                ax.text(x[i], medians[i]+delta, 
                        '%0.2f'%medians[i], 
                        fontsize=10,
                        horizontalalignment='center',
                        color=aspect[nql-1]['facecolor'])

        if print_number:
            delta = y0-ax.get_ylim()[0]
            for i in range(len(numbers)):
                ax.text(x[i], y0-delta/2, 
                        '(%d)'%numbers[i], 
                        fontsize=8,
                        horizontalalignment='center')


