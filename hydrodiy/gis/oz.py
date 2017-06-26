''' Module to plot data on an Australia map '''

import re, os, tarfile
import pkg_resources

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

from mpl_toolkits import basemap

# Decompress australia shoreline shapefile
F_HYGIS_DATA = pkg_resources.resource_filename(__name__, 'data')
FSHP_COAST = os.path.join(F_HYGIS_DATA, 'australia_coastline_simplified.shp')
FSHP_DRAINAGE = os.path.join(F_HYGIS_DATA, 'drainage_divisions_lines_simplified.shp')

if not os.path.exists(FSHP_COAST):
    tar = tarfile.open(re.sub('shp', 'tar.gz', FSHP_COAST))
    for item in tar:
        tar.extract(item, F_HYGIS_DATA)

REGIONS = ['CAPEYORK', 'AUS', 'COASTALNSW', \
                    'MDB', 'VIC+TAS', 'PERTH', 'QLD']


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

        # Read shape
        nm = 'drainage'
        self._map.readshapefile(re.sub('\\.shp$', '', FSHP_DRAINAGE),
                        nm, drawbounds=False)

        # Loop through shapes
        shapes = getattr(self._map, nm)
        for shape in shapes:
            x, y = zip(*shape)
            self._map.plot(x, y, marker=None, *args, **kwargs)


    def drawcoast(self, hires=False, *args, **kwargs):
        ''' plot coast line

            Parameters
            -----------
            hires : bool
                Use high resolution  boundary shapfile
                FEATURE DISABLED AT THE MOMENT
        '''

        # Avoids confusion between hires and other arguments
        if not isinstance(hires, bool):
            raise ValueError('hires is not a boolean')

        if hires:
            self.drawpolygons(re.sub('.shp', '', FSHP_COAST), *args, **kwargs)
        else:
            self._map.drawcoastlines(*args, **kwargs)


    def drawrelief(self, *args, **kwargs):
        ''' plot shaded relief map '''

        self._map.shadedrelief(*args, **kwargs)


    def drawstates(self, *args, **kwargs):
        ''' plot states boundaries '''
        self._map.drawstates(*args, **kwargs)


    def drawpolygons(self, fshp, *args, **kwargs):
        ''' Draw polygon shapefile. Arguments sent to PatchCollection constructor  '''
        nm = os.path.basename(fshp)
        self._map.readshapefile(fshp, nm, drawbounds = False)

        patches = []
        for shape in getattr(self._map, nm):
            patches.append(Polygon(np.array(shape), True))

        self.ax.add_collection(PatchCollection(patches, *args, **kwargs))


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


    def set_lim_region(self, region):
        ''' Set lat/lon box for specific regions in Australia

        Parameters
        -----------
        region : str
            Region name. See hydrodiy.gis.oz.REGIONS

        Returns
        -----------
        xlim : list
            Xmin/Xmax bounds

        ylim : list
            Ymin/Ymax bounds
        '''

        if region == 'CAPEYORK':
            xlim = [137., 148.7]
            ylim = [-24.4, -10.]

        elif region == 'AUS':
            xlim = [109., 155]
            ylim = [-44.4, -9.]

        elif region == 'COASTALNSW':
            xlim = [147.5, 155.]
            ylim = [-38.5, -29.9]

        elif region == 'MDB':
            xlim = [138., 155.]
            ylim = [-40.6, -23.]

        elif region == 'VIC+TAS':
            xlim = [136., 151.]
            ylim = [-44., -33.]

        elif region == 'PERTH':
            xlim = [107., 126.]
            ylim = [-44., -37.]

        elif region == 'QLD':
            xlim = [135., 155.]
            ylim = [-29., -9.]

        else:
            raise ValueError(('Region {0} not recognised, ' + \
                    'should be in {1}').format(region, \
                    '/'.join(REGIONS)))

        self.set_lim(xlim, ylim)

