
import matplotlib.pyplot as plt

try:
    from mpl_toolkits import basemap
    has_basemap = True

except ImportError:
    has_basemap = False

class Oz:
    ''' Plot Australia coast lines and state boundaries '''

    def __init__(self, ax=None, 
        ulat=-9., llat=-43., llon=108., rlon=151.5,
        resolution='l', 
        remove_axis = True):

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

    def drawcoast(self):
        ''' plot coast line '''
        self.map.drawcoastlines()

    def drawrelief(self):
        ''' plot shaded relief map '''
        self.map.shadedrelief()

    def drawstates(self, linestyle='--'):
        ''' plot states boundaries '''
        self.map.drawstates(linestyle=linestyle)

    def plot(self, long, lat, *args, **kwargs):
        ''' Plot points in map '''

        if len(long) != len(lat):
            raise ValueError('len(long) (%d) != len(lat) (%d)' % (len(long), len(lat)))

        x, y = self.map(long, lat)
        self.map.plot(x, y, *args, **kwargs)

             

