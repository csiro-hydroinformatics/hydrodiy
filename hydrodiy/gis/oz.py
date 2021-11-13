''' Module to plot data on an Australia map '''

import re, os, json, tarfile
import pkg_resources

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class HYGisOzError(Exception):
    pass

HAS_BASEMAP = False
try:
    from mpl_toolkits import basemap
    HAS_BASEMAP = True
except (ImportError, FileNotFoundError) as err:
    pass

HAS_PYSHP = False
try:
    import shapefile
    HAS_PYSHP = True
except (ImportError, FileNotFoundError) as err:
    pass


# Decompress australia shoreline shapefile
FDATA = pkg_resources.resource_filename(__name__, 'data')
FCOAST10 = os.path.join(FDATA, 'ne_10m_admin_0_countries_australia.shp')
FCOAST50 = os.path.join(FDATA, 'ne_50m_admin_0_countries_australia.shp')
FSTATES50 = os.path.join(FDATA, 'ne_50m_admin_1_states_australia.shp')
FDRAINAGE = os.path.join(FDATA, 'drainage_divisions_lines_simplified.shp')

# Lat long coordinate boxes for regions in Australia
freg = os.path.join(FDATA, 'regions.json')
with open(freg, 'r') as fo:
    REGIONS = json.load(fo)

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
        if not HAS_BASEMAP:
            raise HYGisOzError('Basemap package could not be imported')

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



def ozlayer(ax, name, filter_field=None, filter_regex=None, proj=None, \
                fixed_lim=True, *args, **kwargs):
    ''' plot Australian geographic layer in axes using data
    from Natural Earth.
    (see https://www.naturalearthdata.com/)

    Parameters
    -----------
    ax : matplotlib.axes
        Axe to draw data on
    name : str
        Name of the layer. Data available are:
        - ozcoast10m, ozcoast50m: Natural Earth coast line at
            10m and 50m resolution respectively.
        - states50m : Natural Earth state boundaries at 50m
            resolution.
        - drain: Drainage boundaries from Geofabric.
    filter_field : str
        Shapefile field to filter on.
    fiter_regex : str
        Regular expression to filter from filtered field.
    proj : pyproj.projProj
        Map projection. Example with transform to GDA94:
        proj = pyproj.Proj('+init=EPSG:3112')
    args, kwargs
        Arguments passed to matplotlib.axes.plot function.
    '''
    if not HAS_PYSHP:
        raise HYGisOzError('pyshp package could not be imported')

    # Select shapefile to draw
    if name == 'ozcoast10m':
        fshp = FCOAST10
    elif name == 'ozcoast50m':
        fshp = FCOAST50
    elif name == 'drainage':
        fshp = FDRAINAGE
    elif name == 'states50m':
        fshp = FSTATES50
    else:
        if os.path.exists(name):
            fshp = name
        else:
            raise HYGisOzError('Expected name in '+\
                '[ozcoast50m|ozcoast110m|states50m|drainage],'+\
                ' got {}'.format(name))

    # Select axis
    if ax is None:
        ax = plt.gca()

    # Plotting function
    def plotit(x, y, recs):
        # Project
        if not proj is None:
            x, y = np.array([proj(xx, yy) for xx, yy in zip(x, y)]).T

        # Plot
        lines = ax.plot(x, y, '-', *args, **kwargs)

        # Return line object
        return (recs, lines[-1])

    # Get xlim and ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot shapefile
    with shapefile.Reader(fshp) as shp_object:
        # Get shapefile fields
        fields = np.array([f[0] for f in shp_object.fields])[1:]

        if not filter_field is None:
            # Filter field
            if not filter_field in fields:
                raise HYGisOzError('Expected filter_field in '+\
                    '/'.join(list(fields)) + ', got '+filter_field)

            ifilter = np.where(fields == filter_field)[0][0]
        else:
            ifilter = None

        # Draw polygons
        lines = []
        for shp, rec in zip(shp_object.shapes(), shp_object.records()):
            # Apply filter if needed
            if not ifilter is None:
                if not re.search(filter_regex, rec[ifilter]):
                    continue

            # Records to series
            recs = pd.Series(rec, index=fields)

            # Get polygon coordinates
            x, y = np.array(shp.points).T

            if hasattr(shp, 'parts'):
                if len(shp.parts) > 1:
                    starts = np.array(shp.parts)
                    ends = np.append(np.array(shp.parts)[1:], len(x))

                    # Plot each part separately
                    for ipart in range(len(shp.parts)):
                        i1, i2 = starts[ipart], ends[ipart]
                        lines.append(plotit(x[i1:i2], y[i1:i2], recs))
                else:
                    # plot
                    lines.append(plotit(x, y, recs))
            else:
                # plot
                lines.append(plotit(x, y, recs))

    if fixed_lim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return lines
