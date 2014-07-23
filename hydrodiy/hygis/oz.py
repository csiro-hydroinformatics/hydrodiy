import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from hygis import gutils

class Oz:
    ''' retrieve Australia coast lines and state boundaries '''

    def __init__(self):
        FHYGIS, ozfile = os.path.split(__file__)
        polygons_file = os.path.join(FHYGIS, 'data',
                                    'oz_data.csv')
        self.polygons_xy = pd.read_csv(polygons_file)

        centroids_file = os.path.join(FHYGIS, 'data',
                                    'oz_states_centroids.csv')
        self.centroids_xy = pd.read_csv(centroids_file, 
                                    index_col='state')
        self.reliefmap_file = os.path.join(FHYGIS, 'data',
                                    'australia.png')

    def get_polygon_xy(self, item):
        idx = self.polygons_xy.item==item
        return self.polygons_xy.ix[idx]

    def get_range(self, item=None):
        ''' Get x/y range for a map item (e.g. QLD)'''
        if item is None:
            return (112., 154., -45., -8.)
        else:
            xy = self.get_polygon_xy(item)
            mini = xy.min()
            maxi = xy.max()
            return (mini['x'], maxi['x'], 
                    mini['y'], maxi['y'])

    def set_rangeoz(self, ax, turn_off_axis=True):
        ''' Get nice x/y range for Australian coastline '''
        ax.axis('equal')
        ax.set_xlim((112.0, 154.0))
        ax.set_ylim((-45.0, -8.0))
        if turn_off_axis:
            ax.axis('off')

    def plot_items(self, ax, items=['coast_mainland','TAS']):
        ''' plot a list of items (e.g. QLD, TAS)'''
        if type(items) is not list:
            items = [items]
        for it in items:
            xy = self.get_polygon_xy(it)
            ids = np.unique(xy.id)
            for id in ids:
                idx = xy.id==id
                ax.plot(xy.x[idx], xy.y[idx], 'k-')

    def plot_coast(self, ax):
        ''' plot coast line '''
        self.plot_items(ax, ['TAS', 'coast_mainland'])

    def plot_ozpng(self, ax):
        ''' plot shaded relief map '''
        gutils.plot_geoimage(ax, self.reliefmap_file)

    def plot_states(self, ax):
        ''' plot states boundaries '''
        states = ['NSW', 'QLD', 'NT', 'VIC', 'WA', 'TAS', 'SA']
        self.plot_items(ax, states)

    def plot_states_data(self, ax, data, fontsize=10):
        ''' plot data for each state in the plot margin 
            data is a pandas.Series with state name
            as index (NSW, ACT, ...)
        '''
        for state, value in data.iteritems():
            xy = self.centroids_xy.loc[state,:]    
            x1 = xy['xcenter']
            y1 = xy['ycenter']
            x2 = xy['xoutside']
            y2 = xy['youtside']
            ha = xy['ha']
            va = xy['va']
            ax.annotate(value, 
                    arrowprops={'edgecolor':'grey',
                        'facecolor':'grey', 'width':1,
                        'headwidth':4}, size=fontsize,
                    bbox={'facecolor':'white', 'edgecolor':'white'},
                    va=va, ha=ha,
                    xy=(x1, y1), xytext=(x2, y2))

    def plot_mask(self, ax, 
            facecolor='white', 
            alpha=1.0):
        ''' plot the coastline mask '''
        for m in ['mask_mainland','mask_tas']:
            xy = self.get_polygon_xy(m)
            nval = len(xy)
            verts = zip(xy.x, xy.y)
            codes = [Path.LINETO]*nval
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            path = Path(verts, codes)
            patch = patches.PathPatch(path, 
                        facecolor=facecolor, 
                        edgecolor=facecolor,
                        alpha=alpha)
            ax.add_patch(patch)



