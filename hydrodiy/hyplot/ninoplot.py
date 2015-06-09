# -*- coding: utf-8 -*-

import os, json

import datetime
from dateutil.relativedelta import relativedelta as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

from hyplot import putils

# Read nino periods
FROOT = os.path.abspath(os.path.dirname(__file__))

fn = '%s/ninoplot_events.json' % FROOT
enso_config = json.load(open(fn,'r'))

def get_ninots():

    nino_ts1 = []
    nino_ts2 = pd.Series(0, 
            index=pd.date_range('1950-01-01', '2020-01-01', freq='MS'))
    
    for e in enso_config['nino']:
        ev = enso_config['nino'][e]
        d1 = pd.to_datetime(ev['start'])
        d2 = pd.to_datetime(ev['end'])
    
        nino_ts1.append({'time':d1, 'value':0})
        nino_ts1.append({'time':d1, 'value':1})
        nino_ts1.append({'time':d2, 'value':1})
        nino_ts1.append({'time':d2, 'value':0})
    
        idx = nino_ts2.index >= d1
        idx = idx & (nino_ts2.index <= d2)
        nino_ts2[idx] = 1
    
    nino_ts1 = pd.DataFrame(nino_ts1)
    nino_ts2.name = 'nino'
    
    return nino_ts1, nino_ts2

def get_nmonth_fromstart(s, start):

    nm =  np.array([dt(x,start).months+dt(x,start).years*12 
                    for x in s])

    return nm

def smooth(data, nmonth_smooth):

    data2 = data.fillna(method='pad')

    data2s = pd.rolling_mean(data2, nmonth_smooth).shift(-nmonth_smooth/2)

    data2s = data2s.fillna(method='pad')

    return data2s


class NinoPlot:
    
    def __init__(self,
            start = datetime.datetime(1970, 1, 1),
            color_background=putils.bureau_background_color,
            color_text = 'white',
            color_spines = 'white',
            font_size = 18,
            fig=None):

        self.start = start

        self.color_background = color_background
        self.color_text = color_text
        self.color_spines = color_spines

        # Set matplotlib
        mpl.rcdefaults()
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['legend.shadow'] = True
        
        mpl.rcParams['axes.labelcolor'] = self.color_text 
        mpl.rcParams['xtick.color'] = self.color_text
        mpl.rcParams['ytick.color'] = self.color_text
        mpl.rcParams['text.color'] = self.color_text

        self.set_font()
       
        if fig is None:
            self.fig = plt.figure()

        self.gs = gridspec.GridSpec(2,2, height_ratios=[1, 3],
                        width_ratios=[17,1])

        ts1, ts2 = get_ninots()
        self.nino_ts1 = ts1
        self.nino_ts2 = ts2

    def _set_spines_color(self, ax):

        for bn in ['left', 'bottom', 'top', 'right']:
            ax.spines[bn].set_color(self.color_spines)


    def _draw_legend(self, ax):
         
        leg = ax.legend(loc=2)

        try:
            leg.get_frame().set_alpha(0.5)

            for text in leg.get_texts(): 
                text.set_color('black')
        except:
            pass


    def _set_xtick(self, ax, startplot, endplot):

        xxtxt = pd.date_range(datetime.datetime(startplot.year+1, 
                    startplot.month, 1), 
                    endplot, freq='2AS')

        xxl = get_nmonth_fromstart(xxtxt, self.start)

        ax.set_xticks(xxl)

        ax.set_xticklabels([d.strftime('%Y') for d in xxtxt])
        
    
    def _set_decoration(self, ax, 
            startplot, endplot, ylim,
            legend, title, ylabel, 
            yticks=[], 
            ygrid=False):

        ax.set_xlim(get_nmonth_fromstart([startplot, endplot], self.start))
        
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)

        ax.set_title(title) 

        if legend:
            self._draw_legend(ax)
            
        self._set_xtick(ax, startplot, endplot)

        self._set_spines_color(ax)

        ax.set_yticks(yticks)

        if ygrid:
            ax.yaxis.grid(True, color=self.color_spines)


    def set_font(self,
            family='sans-serif', 
            weight='medium',
            size=18, size_legend=16):

        self.font = {'family' : family,
                'weight' : weight,
                'size'   : size}
        
        mpl.rc('font', **self.font)

        mpl.rcParams['legend.fontsize'] = size_legend


    def savefig(self, filename):

        self.fig.set_size_inches((21,10))

        self.fig.tight_layout()
        
        self.fig.savefig(filename, facecolor=self.color_background)



    def toplot(self, data, 
            ylim,
            startplot = datetime.datetime(1980, 1, 1),
            endplot = datetime.datetime(2015, 4, 1),
            nmonth_smooth = 6,
            color = 'white',
            line_width = 0.8,
            line_width_smooth = 3,
            ylabel = '', 
            yticks = [], 
            ygrid = False,
            title = '',
            legend = True):

        # Get ax
        ax = plt.subplot(self.gs[0,0], axisbg=self.color_background)

        # Plot data
        xx = get_nmonth_fromstart(data.index, self.start)

        ax.plot(xx, data.values, color, lw=line_width,
            label=data.name) 

        # plot smooth data
        if not nmonth_smooth is None:
            datas = smooth(data, nmonth_smooth)

            ax.plot(xx, datas.values, '-', 
                color=color, 
                lw=line_width_smooth,
                label = '%d month av.' % nmonth_smooth) 

        # Nino periods
        xx = get_nmonth_fromstart(self.nino_ts1['time'], self.start)

        yy = ylim[0] + (ylim[1]-ylim[0]) * self.nino_ts1['value']

        ax.fill_between(xx, ylim[0]*np.ones_like(yy), 
                yy, facecolor='red', alpha=0.7)

        ax.plot([], [], color='red', lw=10, label=u'El Niño events') 

        # Decorations
        self._set_decoration(ax, startplot, endplot, 
            ylim, legend, title, ylabel, yticks, ygrid)


    def bottomplot_bars(self,
            data,
            colors,
            ylim,
            startplot = datetime.datetime(1980, 1, 1),
            endplot = datetime.datetime(2015, 4, 1),
            title = '',
            ylabel = '',
            yticks = [], 
            ygrid = False,
            barwidth = 1,
            hide_non_nino=False,
            legend = True):

        if data.shape[1] != len(colors):
            raise ValueError(('if data.shape[1](%d) != len(colors)(%d)') % (
                data.shape[1], len(colors)))

        # Get ax
        ax = plt.subplot(self.gs[1,0], axisbg=self.color_background)

        # Add nino event
        tmp = data.join(self.nino_ts2)
        data0 = data[tmp['nino']==0]
        data1 = data[tmp['nino']==1]

        means = data.mean()
        if hide_non_nino:
            means = data1.mean()

        # Get abscissae
        xx0 = get_nmonth_fromstart(data0.index, self.start)
        xx1 = get_nmonth_fromstart(data1.index, self.start)

        # Bar width (1 month)
        barwidth = 1
        
        bottom0 = data0.iloc[:, 0].values * 0.
        bottom1 = data1.iloc[:, 0].values * 0.
        
        labels = list(data.columns)
        b0 = []
        b1 = []
        
        for i in range(data.shape[1]):
            
            alpha = None
            barwidth0 = barwidth

            if hide_non_nino:
                alpha = 0.4
                barwidth0 = barwidth * 1.1

            # plot Non nino years
            bb0 = ax.bar(xx0, data0.iloc[:, i].values, 
                width=barwidth0, alpha=alpha,
                bottom=bottom0, color=colors[i], edgecolor='none')

            b0.append(bb0)
            bottom0 += data0.iloc[:, i].values

            # Plot nino years
            bb1 = ax.bar(xx1, data1.iloc[:, i].values, 
                width=barwidth, 
                bottom=bottom1, color=colors[i], edgecolor='none')
            
            b1.append(bb1)
            bottom1 += data1.iloc[:, i].values

            ax.plot([], [], color=colors[i], lw=10, label=labels[i]) 


        ax.tick_params(length=0.01)

        # Decorations
        self._set_decoration(ax, startplot, endplot, ylim, 
            legend, title, ylabel, yticks, ygrid)

        return means


    def bottomplot_line(self,
            data,
            ylim,
            startplot = datetime.datetime(1980, 1, 1),
            endplot = datetime.datetime(2015, 4, 1),
            idx_highlight = None,
            color = 'white',
            color_nino = 'red',
            label_highlight = '',
            line_width = 0.8,
            line_width_highlight = 4,
            line_width_nino = 6,
            title = '',
            ylabel = '',
            yticks = [], 
            ygrid = False,
            barwidth = 1,
            hide_non_nino=False,
            legend = True):

        # Get ax
        ax = plt.subplot(self.gs[1,0], axisbg=self.color_background)

        # Get x values
        xx = get_nmonth_fromstart(data.index, self.start)
        
        # Remove non nino years
        means = data.mean()

        if hide_non_nino:

            tmp = pd.DataFrame(data).join(nino_ts2)
            data2 = data.copy()
            data2[tmp['nino']==0] = np.nan
            ax.plot(xx, data2.values, lw=line_width_nino, color=color_nino)
           
            means = data2.mean()

        if not idx_highlight is None:

            data2 = data.copy()
            data2[~idx_highlight] = np.nan

            ax.plot(xx, data2.values, 
                lw=line_width_highlight, 
                color='white', 
                label=label_highlight)

            means = data2.mean()

        # Plot line
        ax.plot(xx, data.values, '-', 
            lw=line_width, 
            color=self.color_text)

        # Decorations
        self._set_decoration(ax, startplot, endplot, ylim, 
            legend, title, ylabel, yticks, ygrid)

        return means


    def bottomplot_average(self,
            ylim, 
            colors,
            title, 
            label,
            means = [], 
            barwidth = 1,
            font_size_large = 20,
            font_size_small = 16):

        # Get ax
        ax = plt.subplot(self.gs[1,1], axisbg=self.color_background)

        ax.set_ylim(ylim)

        if len(means)>0:

            if len(means) != len(colors):
                raise ValueError('len(means)(%d) != len(colors)(%d)' % (
                    len(means), len(colors)))

            # Bar plots
            bottom = 0.
            b = []
    
            for i in range(len(means)):
                
                bb = ax.bar(0, means.iloc[i], width=barwidth, 
                    bottom=bottom, color=colors[i], edgecolor='none')

                ax.text(0.5, bottom+means.iloc[i]/2, '%0.0f%%' % means.iloc[i], 
                        ha='center', fontsize=font_size_large)
                
                b.append(bb)

                bottom += means.iloc[i]

        else:
            # Line plot
            ax.text(0.5, np.mean(ylim), title,
                    ha='center', fontsize=font_size_small)

            ax.text(0.5, np.mean(ylim)*0.8, label,
                    ha='center', fontsize=font_size_large)

        ax.set_yticks([])
        ax.set_xticks([])
        self._set_spines_color(ax)
        ax.set_title(title) 
