import re, os, tarfile

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

try:
    from mpl_toolkits import basemap
    has_basemap = True

except ImportError:
    has_basemap = False


# Decompress australia shoreline shapefile
FDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
fshp_coastoz = os.path.join(FDATA, 'australia_coastline_simplified.shp')
fshp_drainageoz = os.path.join(FDATA, 'drainage_divisions_lines_simplified.shp')

if not os.path.exists(fshp_coastoz):
    tar = tarfile.open(re.sub('shp', 'tar.gz', fshp_coastoz))
    for item in tar:
        tar.extract(item, FDATA)


class Oz:

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

        if has_basemap:
            self.map = basemap.Basemap(self.llon, self.llat, self.rlon, self.ulat,
                lat_0=24.75, lon_0=134.0, lat_1=-10, lat_2=-40,
                rsphere=(6378137.00,6356752.3142),
                projection='lcc', resolution=resolution,
                area_thresh=1000, suppress_ticks=True, ax = self.ax)
        else:
            raise ImportError('matplotlib - basemap is not available')

        if remove_axis:
            self.ax.axis('off')

    def get_map(self):
        return self.map

    def get_range(self):
        ''' Get x/y range for the map '''

        return (self.llon, self.rlon, self.ulat, self.llat)

    def set_axrange(self):
        ''' Get nice x/y range for Australian coastline '''
        self.ax.set_xlim((self.llon, self.rlon))
        self.ax.set_ylim((self.ulat, self.llat))


    def drawcoastoz(self, *args, **kwargs):
        ''' plot coast line for Australia only'''

        self.drawpolygons(re.sub('.shp', '', fshp_coastoz), *args, **kwargs)


    def drawdrainageoz(self, *args, **kwargs):
        ''' plot drainage divisions for Australia only'''

        # Read shape
        nm = 'drainage'
        self.map.readshapefile(re.sub('\\.shp$', '', fshp_drainageoz),
                        nm, drawbounds=False)

        # Loop through shapes
        shapes = getattr(self.map, nm)
        for shape in shapes:
            x, y = zip(*shape)
            self.map.plot(x, y, marker=None, *args, **kwargs)


    def drawcoast(self, *args, **kwargs):
        ''' plot coast line '''
        self.map.drawcoastlines(*args, **kwargs)


    def drawrelief(self, *args, **kwargs):
        ''' plot shaded relief map '''

        self.map.shadedrelief(*args, **kwargs)


    def drawstates(self, *args, **kwargs):
        ''' plot states boundaries '''
        self.map.drawstates(*args, **kwargs)


    def drawpolygons(self, fshp, *args, **kwargs):
        ''' Draw polygon shapefile. Arguments sent to PatchCollection constructor  '''
        nm = os.path.basename(fshp)
        self.map.readshapefile(fshp, nm, drawbounds = False)

        patches = []
        for shape in getattr(self.map, nm):
            patches.append(Polygon(np.array(shape), True))

        self.ax.add_collection(PatchCollection(patches, *args, **kwargs))


    def plot(self, long, lat, *args, **kwargs):
        ''' Plot points in map '''

        if len(long) != len(lat):
            raise ValueError(('len(long) (%d) != '
                'len(lat) (%d)') % (len(long), len(lat)))

        x, y = self.map(long, lat)
        self.map.plot(x, y, *args, **kwargs)


    def set_lim(self, xlim, ylim):
        ''' Set xlim and ylim '''

        xxlim, yylim = self.map(np.sort(xlim),np.sort(ylim))

        self.ax.set_xlim(np.sort(xxlim))
        self.ax.set_ylim(np.sort(yylim))


