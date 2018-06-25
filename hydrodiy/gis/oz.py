''' Module to plot data on an Australia map '''

import re, os, tarfile
import pkg_resources

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from mpl_toolkits import basemap

# Decompress australia shoreline shapefile
FDATA = pkg_resources.resource_filename(__name__, 'data')
FCOAST10 = os.path.join(FDATA, 'ne_10m_admin_0_countries_australia.shp')
FCOAST50 = os.path.join(FDATA, 'ne_50m_admin_0_countries_australia.shp')
FDRAINAGE = os.path.join(FDATA, 'drainage_divisions_lines_simplified.shp')

# Lat long coordinate boxes for regions in Australia
REGIONS = {\
    'AUS': { \
        'name' : 'Australia', \
        'xlim' : [109., 155], \
        'ylim' : [-44.4, -9.] \
    }, \
    # States
    'QLD': {\
        'name': 'Queensland', \
        'xlim' : [136.5, 155.], \
        'ylim' : [-30., -9.]\
    }, \
    'TAS':{\
        'name': 'Tasmania',
        'xlim': [144.38, 148.55], \
        'ylim': [-43.83, -40.21] \
    }, \
    'WA':{\
        'name': 'Western Australia', \
        'xlim': [113.52, 129.26], \
        'ylim': [-44., -13.2] \
    }, \
    'NSW':{\
        'name': 'New South Wales', \
        'xlim': [140.6, 154.], \
        'ylim': [-38., -27.5] \
    }, \
    'VIC':{\
        'name': 'Victoria', \
        'xlim': [140.88, 150.68], \
        'ylim': [-39.5, -33.9] \
    }, \
    'NT':{\
        'name': 'Northern Territory', \
        'xlim': [128., 138.40], \
        'ylim': [-26.08, -10.49] \
    }, \
    # Drainage divisions
    'MDB': {\
        'name': 'Murray-Darling drainage division', \
        'xlim' : [138., 155.], \
        'ylim' : [-38.5, -23.] \
    }, \
    'SWC':{\
        'name': 'South-West Coast drainage division',
        'xlim': [113.52, 124.], \
        'ylim': [-35.97, -27.22] \
    }, \
    'PG':{\
        'name': 'Pilbara-Gascoyne drainage division', \
        'xlim': [112., 122.], \
        'ylim': [-30.17, -19.58] \
    }, \
    'NEC':{\
        'name': 'North-East Coast drainage division', \
        'xlim': [142.27, 153.98], \
        'ylim': [-30., -9.33] \
    }, \
    'SEN':{\
        'name': 'South East Coast (NSW) drainage division', \
        'xlim': [148.78, 154.30], \
        'ylim': [-37.70, -27.49] \
    }, \
    'SEV':{\
        'name': 'South East Coast (VIC) drainage division', \
        'xlim': [138.87, 149.84], \
        'ylim': [-39.18, -35.55] \
    }, \
    # Miscellaneous regions
    'SEAUS':{\
        'name': 'South East Australia', \
        'xlim': [140.81, 153.6], \
        'ylim': [-44.03, -28.3] \
    }, \
    'NA':{\
        'name': 'Northern Australia', \
        'xlim': [109, 154], \
        'ylim': [-24, -9.33] \
    }, \
    'MDBS': {\
        'name': 'South of Murray Darling Basin',
        'xlim': [138.36, 150.73], \
        'ylim': [-38.6, -31.55] \
    }, \
    'MDBN': {\
        'name': 'North of Murray Darling Basin',
        'xlim': [141.14, 152.78], \
        'ylim': [-33.02, -23.] \
    }, \
    'BIDGEE':{\
        'name': 'Murrumbidgee Basin',
        'xlim' : [142.37, 149.86], \
        'ylim' : [-36.83, -33.5] \
    }, \
    'MURRAY':{\
        'name': 'Murray Basin',
        'xlim': [138.36, 149.86], \
        'ylim': [-38.6, -34.18] \
    }, \
    'CAPEYORK': {\
        'name' : 'Cape York', \
        'xlim' : [137., 148.7], \
        'ylim' : [-17., -10.] \
    }, \
    'COASTALNSW': {\
        'name': 'Coastal New South Wales', \
        'xlim' : [147.5, 155.], \
        'ylim' : [-38.5, -27.8]
    }, \
    'VICTAS': {\
        'name': 'Victoria and Tasmania', \
         'xlim': [140.5, 151.], \
         'ylim': [-44., -33.] \
    }
}



class Oz:
    ''' Class to plot data on an Australia map '''

    def __init__(self, ax=None,
        ulat=-9., llat=-43., llon=108., rlon=151.5,
        resolution='l',
        remove_axis = True):
        '''
        Plot data on Australia map

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw data on
        ulat : float
            Latitude of map upper bound
        llat : float
            Latitude of map lower bound
        llon : float
            Longitude of map left bound
        rlon : float
            Longiture of map right bound
        resolution : string
            Map coastline resolution. See mpl_toolkits.basemap
            'c' : Crude
            'l' : Low
            'i' : Intermediate
            'h' : High
            'f' : Full
        remove_axis : bool
            Hide axis in axe or not

        Example
        -----------
        >>> import numpy as np
        >>> from hygis import oz
        >>> import matplotlib.pyplot as plt
        >>> nval = 200
        >>> x = np.random.uniform(130, 150, nval)
        >>> y = np.random.uniform(-40, -10, nval)
        >>> fig, ax = plt.subplots()
        >>> om = oz.Oz(ax = ax)
        >>> om.drawcoast()
        >>> om.plot(x, y, 'o')

        '''

        self.ulat = ulat
        self.llat = llat
        self.llon = llon
        self.rlon = rlon
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        self._map = basemap.Basemap(self.llon, self.llat, self.rlon, self.ulat,
            lat_0=24.75, lon_0=134.0, lat_1=-10, lat_2=-40,
            rsphere=(6378137.00,6356752.3142),
            projection='lcc', resolution=resolution,
            area_thresh=1000, suppress_ticks=True, ax = self.ax)

        if remove_axis:
            self.ax.axis('off')


    @property
    def map(self):
        return self._map


    def get_range(self):
        ''' Get x/y range for the map '''

        return (self.llon, self.rlon, self.ulat, self.llat)


    def set_axrange(self):
        ''' Get nice x/y range for Australian coastline '''
        self.ax.set_xlim((self.llon, self.rlon))
        self.ax.set_ylim((self.ulat, self.llat))


    def drawdrainage(self, *args, **kwargs):
        ''' plot drainage divisions for Australia only '''

        # Adding default color to drainage
        if not 'color' in kwargs:
            kwargs['color'] = 'k'

        # Read shape
        nm = 'drainage'
        self._map.readshapefile(re.sub('\\.shp$', '', FDRAINAGE),
                        nm, drawbounds=False)

        # Loop through shapes
        shapes = getattr(self._map, nm)
        for shape in shapes:
            x, y = zip(*shape)
            self._map.plot(x, y, marker=None, *args, **kwargs)


    def drawcoast(self, hires=False, edgecolor='black', \
            facecolor='none', alpha=1., \
            linestyle='-', linewidth=1.):
        ''' plot coast line

            Parameters
            -----------
            hires : bool
                Use high resolution  boundary shapfile
            facecolor : str
                Filling color of the Australian continent
            edgecolor : str
                Color of the coast line
            alpha : float
                Transparency in [0, 1]
            linestyle : str
                Line style
            linewidth : float
                Line width
        '''

        if hires:
            self.drawpolygons(re.sub('.shp', '', FCOAST10), \
                facecolor=facecolor, \
                edgecolor=edgecolor, \
                linewidth=linewidth, \
                linestyle=linestyle, \
                alpha=alpha)
        else:
            self.drawpolygons(re.sub('.shp', '', FCOAST50), \
                facecolor=facecolor, \
                edgecolor=edgecolor, \
                linewidth=linewidth, \
                linestyle=linestyle, \
                alpha=alpha)


    def drawrelief(self, *args, **kwargs):
        ''' plot shaded relief map '''

        self._map.shadedrelief(*args, **kwargs)


    def drawstates(self, *args, **kwargs):
        ''' plot states boundaries '''
        self._map.drawstates(*args, **kwargs)


    def drawpolygons(self, shp, \
                facecolor='none', \
                edgecolor='k',\
                linewidth=1., \
                linestyle='-', \
                alpha=1., \
                hatch=None):
        ''' Draw polygon shapefile. Arguments sent to PatchCollection constructor  '''
        nm = os.path.basename(shp)
        self._map.readshapefile(shp, nm, drawbounds = False)

        for shape in getattr(self._map, nm):
            # plot contour
            x, y = zip(*shape)
            self._map.plot(x, y, marker=None, \
                    color=edgecolor, alpha=alpha,\
                    linestyle=linestyle, \
                    linewidth=linewidth)

            # plot interior
            poly = Polygon(np.array(shape), closed=True,\
                    ec='none', \
                    fc=facecolor, \
                    alpha=alpha, \
                    hatch=hatch)
            self.ax.add_patch(poly)

        #pcoll = PatchCollection(patches)
        #self.ax.add_collection(pcoll)


    def plot(self, long, lat, *args, **kwargs):
        ''' Plot points in map '''

        if len(long) != len(lat):
            raise ValueError(('len(long) (%d) != '
                'len(lat) (%d)') % (len(long), len(lat)))

        x, y = self._map(long, lat)
        self._map.plot(x, y, *args, **kwargs)


    def set_lim(self, xlim, ylim):
        ''' Set a lat/lon box range '''

        # Set lim to map
        xxlim, yylim = self._map(np.sort(xlim),np.sort(ylim))

        self.ax.set_xlim(np.sort(xxlim))
        self.ax.set_ylim(np.sort(yylim))


    def set_lim_region(self, region=None):
        ''' Set lat/lon box for specific regions in Australia

        Parameters
        -----------
        region : str
            Region name. See hydrodiy.gis.oz.REGIONS.
            If None, set region to AUS (Australia)
        '''
        if region is None:
            region = 'AUS'

        if region in REGIONS:
            reg = REGIONS[region]
            xlim = reg['xlim']
            ylim = reg['ylim']
        else:
            allregions = '/'.join(list(REGIONS.keys()))
            raise ValueError(('Expected region in {0}, ' +\
                    'got {1}').format(allregions, region))

        self.set_lim(xlim, ylim)

