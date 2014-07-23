from datetime import datetime
import datetime

from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms

import numpy as np
import pandas as pd

from hystat import sutils
import _match 

class RingPlot:

    def __init__(self, fig, nrows, ncols, 
            centreaxe_overlap=0., 
            centreaxe_aspect='auto', 
            centreaxe_axis_off=True,
            ringaxe_axis_off=False,
            ringaxe_border=0.05):
        
        assert ncols>2
        assert nrows>2
        assert (centreaxe_overlap>=0.) & (centreaxe_overlap<1.)

        self.fig = fig
        self._nrows = nrows
        self._ncols = ncols
        self._centreaxe_overlap = centreaxe_overlap
        self._ringaxe_border = ringaxe_border

        ax_loc = np.ones((nrows, ncols))
        ax_loc[1:-1, 1:-1] = 0.
        idx = np.where(ax_loc==1)
        x = idx[1]*1./ncols + 1./ncols/2 
        y = idx[0]*1./nrows + 1./nrows/2

        # set ring axes data 
        ringaxes_data = {'irow':idx[0], 'icol':idx[1], 
                'x_fig':x, 'y_fig':y, 
                'x_ax':0, 'y_ax':0, 'ax':0}
        ringaxes_data = pd.DataFrame(ringaxes_data)
        self.ringaxes = ringaxes_data.set_index(['irow', 'icol'])
        self.ringaxes['irow'] = idx[0]
        self.ringaxes['icol'] = idx[1]

        # Initialise centre axe
        wx = 1./ncols
        wy = 1./nrows
        left = (1-centreaxe_overlap)*(wx+ringaxe_border)
        bottom = (1-centreaxe_overlap)*(wy+ringaxe_border)
        rect = [left, bottom, 1-2*left, 1-2*bottom]
        self.centreaxe = self.fig.add_axes(rect, 
                                    aspect=centreaxe_aspect)
        if centreaxe_axis_off:
            self.centreaxe.axis('off')

        # Initialise ring axes 
        for idx in self.ringaxes.index:
            self._set_ringaxe(idx[0], idx[1], ringaxe_axis_off)

    def _xyfig2ax(self, ax, x_fig, y_fig):
        ''' Convert figure coordinates into axe coordinates '''
        # Get transform functions 
        fig2disp = self.fig.transFigure.transform
        disp2ax =  ax.transData.inverted().transform

        # conversion
        x_disp, y_disp = fig2disp((x_fig, y_fig))
        x_ax, y_ax = disp2ax((x_disp, y_disp))

        return x_ax, y_ax

    def _set_ringaxe(self, irow, icol, ringaxe_axis_off=False):
        ''' 
            Generate one ring axe with location irow and icol 
            and adds coordinates of ring axe in centre axe coordinate
            system
        '''
        x_fig = self.ringaxes.loc[(irow, icol), 'x_fig']
        y_fig = self.ringaxes.loc[(irow, icol), 'y_fig']
        wx = 1./self._ncols
        wy = 1./self._nrows
        rect = [x_fig-wx/2+self._ringaxe_border, 
                y_fig-wy/2+self._ringaxe_border, 
                wx-2*self._ringaxe_border, 
                wy-2*self._ringaxe_border]
        ax = self.fig.add_axes(rect)
        if ringaxe_axis_off:
            ax.axis('off')
        self.ringaxes.loc[(irow, icol), 'ax'] = ax

        # Creates coordinates in central axis
        x_ax, y_ax = self._xyfig2ax(self.centreaxe, x_fig, y_fig)
        self.ringaxes.loc[(irow, icol), 'x_ax'] = x_ax
        self.ringaxes.loc[(irow, icol), 'y_ax'] = y_ax

    def match_pts(self, centreaxe_x, centreaxe_y):
        ''' 
            Use C function to compute best match between 
            points in centre axe (centreax_x, centreax_y) 
            and ring axes. Returns a list of tuple 
            (irow, icol) indicating the mapping for 
            each point.
        ''' 
        match_final = np.zeros((len(centreaxe_x),), np.int32)
        x = self.ringaxes['x_ax'].values
        y = self.ringaxes['y_ax'].values
        _match.match(x, y, centreaxe_x , centreaxe_y, 
                            match_final)

        ir = self.ringaxes['irow'][match_final]
        ic = self.ringaxes['icol'][match_final]
        mapping = [(iir, iic) for iir, iic in zip(ir, ic)]

        return mapping


    def draw_line(self, irow, icol, x, y, 
            marker='o', markersize=10, facecolor='b',
            connection_style=None):
        ''' 
            Draw line between ring ax and centre ax 
            at position (x,y). connection_style can be used to 
            refine line style (e.g. 
               connection_style='angle, angleA=0., angleB=90., rad=5.')
        '''

        # Connection style string used in annotate
        connstyle = 'arc3, rad=0.'
        if not connection_style is None:
            connstyle = connection_style

        # Find coordinates of start of annotation line
        x_fig = self.ringaxes.loc[(irow, icol), 'x_fig']
        y_fig = self.ringaxes.loc[(irow, icol), 'y_fig']

        # Create annotation line
        self.centreaxe.annotate('', xy=(x, y), xycoords='data',
                xytext=(x_fig, y_fig), textcoords='figure fraction',
                arrowprops=dict(arrowstyle='wedge', color='0.5', 
                    lw=1,shrinkA=5, shrinkB=5,
                    connectionstyle=connstyle),
                ha='right', va='top')

        self.centreaxe.scatter(x, y, s=markersize, 
                facecolor=facecolor, marker=marker)

