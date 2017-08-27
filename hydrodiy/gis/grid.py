''' Module containing grid utilities '''

import re, os
import pkg_resources
import math
import copy
import zipfile

from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import c_hydrodiy_gis

# Codes indicating the flow direction for a cell
# i.e.
#  32  64   128       \   |   /
#  16   0     1   =>  <-  x   ->
#   8   4     2       /   |   \
#
FLOWDIRCODE = np.array([[32, 64, 128],
                            [16, 0, 1],
                            [8, 4, 2]]).astype(np.int64)

# Path to hygis data
F_HYGIS_DATA = pkg_resources.resource_filename(__name__, 'data')


class Grid(object):
    ''' Gridded data object '''

    def __init__(self, name, ncols, nrows=None, cellsize=1., \
            xllcorner=0, yllcorner=0, dtype=np.float64, \
            nodata_value=np.nan, \
            comment=''):

        if nrows is None:
            nrows = ncols

        self.name = name
        self.ncols = np.int64(ncols)
        self.nrows = np.int64(nrows)
        self._dtype = dtype
        self.nodata_value = nodata_value

        self.cellsize = np.float64(cellsize)

        self.xllcorner = np.float64(xllcorner)
        self.yllcorner = np.float64(yllcorner)

        self.nodata = np.nan
        self.maxdata = np.inf
        self.mindata = -np.inf
        self.comment = comment

        self._data = np.zeros((self.nrows, self.ncols), dtype=dtype)


    def _getsize(self):
        ''' Returns dimensions of the grid '''
        xll = self.xllcorner
        yll = self.yllcorner
        csz = self.cellsize
        nrows = self.nrows
        ncols = self.ncols
        return xll, yll, csz, nrows, ncols


    def __setitem__(self, index, value):
        ''' Set cell values '''
        index =  np.int64(index)
        self._data.flat[index] = np.array(value).astype(self.dtype)


    def __getitem__(self, index):
        ''' Extract cell values '''
        index =  np.int64(index)
        return self._data.flat[index]


    def __str__(self):
        str = '\nGrid {0}:\n'.format(self.name)
        str += '\tncols    : {0}\n'.format(self.ncols)
        str += '\tnrows    : {0}\n'.format(self.nrows)
        str += '\tcellsize : {0}\n'.format(self.cellsize)
        str += '\txllcorner: {0}\n'.format(self.xllcorner)
        str += '\tyllcorner: {0}\n'.format(self.yllcorner)
        str += '\tdtype    : {0}\n'.format(self.dtype)
        str += '\tno_data_value : {0}\n'.format(self.nodata_value)
        str += '\tcomment  : {0}\n'.format(self.comment)

        return str


    @classmethod
    def from_header(cls, filename):
        ''' Create grid from header file

        Parameters
        -----------
        fileheader : str
            Path to the header file or BIL file

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        '''

        fileheader = re.sub('(bil|flt)$', 'hdr', filename)

        if not os.path.exists(fileheader):
            raise ValueError(('File {0} does not '+
                'exist').format(fileheader))

        config = {
            'name' : re.sub('\\..*$', '', os.path.basename(fileheader)), \
            'xllcorner' : 0., \
            'yllcorner' : 0., \
            'cellsize': 1., \
            'nodata_value' : np.nan, \
            'nbits' : 64, \
            'pixeltype' : 'float', \
            'byteorder' : 'i', \
            'comment': 'No comment'
        }

        # Read property from header
        with open(fileheader, 'r') as fh:
            for line in fh.readlines():
                line = re.split(' ', re.sub(' +', ' ', line))
                pname = line[0].lower()
                try:
                    if pname in ['pixeltype', 'byteorder',
                        'layout', 'comment', 'name']:
                        pvalue = ' '.join(line[1:]).strip().lower()
                    elif pname.startswith('n'):
                        pvalue = int(line[1].strip())
                    else:
                        pvalue = float(line[1].strip())

                    config[pname] = pvalue
                except ValueError:
                    import warnings
                    warnings.warn(('Header field {0} cannot be' + \
                            +'processed').format(pname))
                    pass

        # Define dtype
        if config['byteorder'] == 'm':
            byteorder = '>'
        elif config['byteorder'] == 'i':
            byteorder = '<'
        else:
            raise ValueError('Byteorder {0} not recognised'.format(
                    config['byteorder']))

        pixeltype = re.sub('nsignedint$|^signed|nt|loat', '', config['pixeltype'])
        nbits = config['nbits']//8
        config['dtype'] = np.dtype(byteorder + pixeltype + str(nbits)).type

        # Check cell size / dimensions
        if 'xdim' in config:
            config['cellsize'] = config['xdim']

            # Check the cell are squared (i.e. xdim=ydim)
            if 'ydim' in config:
                if config['ydim'] != config['xdim']:
                    raise ValueError(('xdim {0} != ' +
                        'ydim {1}').format(config['xdim'],
                            config['ydim']))

        if 'ulxmap' in config:
            config['xllcorner'] = config['ulxmap']
            config['yllcorner'] = config['ulymap'] - \
                                config['cellsize']*config['nrows']

        # Filters config data
        keys = ['name', 'ncols', 'nrows', 'cellsize', 'comment', \
            'xllcorner', 'yllcorner', 'dtype', 'nodata_value']
        config = {k:config[k] for k in config if k in keys}

        # Creates grid
        grid = Grid(**config)

        # Reads data if bil file is there
        filedata = re.sub('hdr$', 'bil',fileheader)
        if os.path.exists(filedata):
            grid.load(filedata)

        return grid


    @classmethod
    def from_dict(cls, dic):
        ''' Create grid from dictionary

        Parameters
        -----------
        dic : dict
            Dictionary

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        '''

        # Init argument
        dic2 = {'name':dic['name'], 'ncols':dic['ncols']}

        # Init optional arguments
        for opt in ['nrows', 'cellsize', 'xllcorner', 'yllcorner', \
            'dtype', 'nodata_value', 'comment']:
            if opt in dic:
                dic2[opt] = dic[opt]

        if 'dtype' in dic2:
            dic2['dtype'] = np.dtype(dic2['dtype']).type

        return  Grid(**dic2)


    @property
    def data(self):
        ''' Get grid data '''
        return self._data

    @data.setter
    def data(self, value):
        ''' Set grid data '''

        _value = np.ascontiguousarray(np.atleast_2d(value))

        if _value.ndim != 2:
            raise ValueError('Expected 2d array, got {0} dimensions'.format(\
                                _value.ndim))

        nrows, ncols = _value.shape

        if nrows != self.nrows:
            raise ValueError(('Wrong number of rows:' + \
                ' data has {0}, but expects {1}').format(nrows, self.nrows))

        if ncols != self.ncols:
            raise ValueError(('Wrong number of columns:' + \
                ' data has {0}, but expects {1}').format(ncols, self.ncols))

        self._data = np.clip(_value, self.mindata,
                                self.maxdata).astype(self.dtype)

    @property
    def dtype(self):
        ''' Get data type '''
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        ''' Set data type '''

        self._dtype = value
        self._data = self._data.astype(value)

    def same_geometry(self, grd):
        ''' Check another grid has the same geometry

        Parameters
        -----------
        grd : hydrodiy.gis.grid.Grid
            External grid

        Returns
        -----------
        identical : bool
            Same geometry True/False
        '''

        identical = grd.ncols == self.ncols
        identical = identical & (grd.nrows == self.nrows)
        identical = identical & np.allclose(grd.xllcorner, self.xllcorner)
        identical = identical & np.allclose(grd.yllcorner, self.yllcorner)
        identical = identical & np.allclose(grd.cellsize, self.cellsize)

        return identical



    def load(self, filename):
        ''' Load data from file

        Parameters
        -----------
        filename : str
            Path to the binary data (only BIL file format at the moment)

        '''

        if not filename.endswith('bil'):
            raise ValueError('Only bil format recognised')

        if not os.path.exists(filename):
            raise ValueError('File {0} does not exist'.format(filename))

        data = np.fromfile(filename, self.dtype)

        nval = self.nrows * self.ncols
        if len(data) != nval:
            raise ValueError(('File contains {0} data points' +
                ', expecting {1}').format(len(self._data), nval))

        self._data = np.clip(data.reshape((self.nrows, self.ncols)),
                        self.mindata, self.maxdata).astype(self.dtype)


    def to_dict(self):
        ''' Export grid metadata to json '''
        js = { \
            'name': self.name, \
            'ncols': self.ncols, \
            'nrows': self.nrows, \
            'cellsize': self.cellsize, \
            'xllcorner': self.xllcorner, \
            'yllcorner': self.yllcorner, \
            'dtype': np.dtype(self.dtype).str, \
            'nodata_value': str(self.nodata_value), \
            'comment': self.comment
        }

        return js


    def save(self, filename):
        ''' Save data as a BIL file '''

        if not filename.endswith('bil'):
            raise ValueError(('Filename ({0}) should end with a' + \
                ' bil extension').format(filename))

        # Print header file
        fileheader = re.sub('bil$', 'hdr', filename)
        with open(fileheader, 'w') as fh:
            for attname in ['nrows', 'ncols', 'xllcorner',
                'yllcorner', 'cellsize']:
                attval = getattr(self, attname)
                fh.write('{0:<14} {1}\n'.format(attname.upper(), attval))

            # nbits
            ddtype = np.dtype(self.dtype)
            nbits = ddtype.itemsize*8
            fh.write('{0:<14} {1}\n'.format('NBITS', nbits))

            # pixel type
            name = re.sub('[0-9]+$', '', ddtype.name)
            if name == 'int':
                pixeltype = 'signedint'
            elif name == 'uint':
                pixeltype = 'unsignedint'
            elif name == 'float':
                pixeltype = 'float'
            else:
                raise ValueError('Type name {0} unrecognised'.format(name))

            fh.write('{0:<14} {1}\n'.format('PIXELTYPE', pixeltype.upper()))

            # Byte order
            if ddtype.byteorder == '>':
                byteorder = 'M'
            else:
                byteorder = 'I'
            fh.write('{0:<14} {1}\n'.format('BYTEORDER', byteorder))

            # Name
            fh.write('{0:<14} {1}\n'.format('NAME', self.name))

            # Comment
            comment = self.comment
            if comment == '':
                comment = 'No comment'
            fh.write('{0:<14} {1}\n'.format('COMMENT', comment))

        # Print data
        self._data.tofile(filename)


    def fill(self, value):
        ''' Initialise grid value

        Parameters
        -----------
        value : float
            Value used to fill the grid data.

        '''
        dtype = self._dtype
        self.data.fill(dtype(value))


    def coord2cell(self, xycoords):
        ''' Return cell number from coordinates.
            Cells are counted from the top left corner, row by row
            starting from 0.

        Example for a 4x4 grid:
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

        Parameters
        -----------
        xycoords : numpy.ndarray
            2D array with x coords in first column and
            y coords in second column.

        Returns
        -----------
        idxcell : numpy.ndarray
            1D array containing cell numbers

        '''
        xll, yll, csz, nrows, ncols = self._getsize()

        xycoords = np.ascontiguousarray(np.atleast_2d(xycoords), dtype=np.float64)
        idxcell = np.zeros(len(xycoords)).astype(np.int64)

        ierr = c_hydrodiy_gis.coord2cell(nrows, ncols, xll, yll,
                                        csz, xycoords, idxcell)
        if ierr>0:
            raise ValueError('c_hydrodiy_gis.coord2cell returns '+str(ierr))

        return idxcell


    def cell2coord(self, idxcells):
        ''' Return coordinate from cell number
            Cells are counted from the top left corner, row by row
            starting from 0.

        Example for a 4x4 grid with (xll,yll)=(0,0):
         0  1  2  3        (0, 3)  (1, 3) (2, 3) (3, 3)
         4  5  6  7   ->   (0, 2)  (1, 2) (2, 2) (3, 2)
         8  9 10 11        (0, 1)  (1, 1) (2, 1) (3, 1)
        12 13 14 15        (0, 0)  (1, 0) (2, 0) (3, 0)

        Parameters
        -----------
        idxcell : numpy.ndarray
            1D array containing cell numbers

        Returns
        -----------
        xycoords : numpy.ndarray
            2D array with x coords in first column and
            y coords in second column.

        '''
        xll, yll, csz, nrows, ncols = self._getsize()

        idxcells = np.ascontiguousarray(np.atleast_1d(idxcells), dtype=np.int64)
        xycoords = np.zeros((len(idxcells), 2)).astype(np.float64)

        ierr = c_hydrodiy_gis.cell2coord(nrows, ncols, xll, yll,
                                        csz, idxcells, xycoords)
        if ierr>0:
            raise ValueError('c_hydrodiy_gis.cell2coord returns '+str(ierr))

        return xycoords


    def neighbours(self, idxcell):
        ''' Compute the codes of cells surrounding a cell.
            Output is a vector of length 8 corresponding to
            0 1 2
            3 X 4 -> [n(0), n(1), ..., n(7)]
            5 6 7

        Parameters
        -----------
        idxcell : int
            cell number

        Returns
        -----------
        neighbour : numpy.ndarray
            1D array containing the 9 neighbouring cell number

        '''
        _, _, _, nrows, ncols = self._getsize()

        idxcell = np.int64(idxcell)
        neighbours = np.zeros(9).astype(np.int64)

        ierr = c_hydrodiy_gis.neighbours(nrows, ncols, idxcell,
                                        neighbours)
        if ierr>0:
            raise ValueError('c_hydrodiy_gis.neighbours returns '+str(ierr))

        return neighbours


    def slice(self, xyslice):
        ''' Extract a profile from the grid

        Parameters
        -----------
        xyslice : numpy.ndarray
            2D array with x coords in first column and
            y coords in second column.

        Returns
        -----------
        zslice : numpy.ndarray
            1D array containing sliced values from gridded data
        '''
        xll, yll, csz, _, _ = self._getsize()

        xyslice = np.ascontiguousarray(np.atleast_2d(xyslice),
                        dtype=np.float64)
        zslice = np.zeros(len(xyslice)).astype(np.float64)
        ierr = c_hydrodiy_gis.slice(xll, yll, csz, self._data,
                    xyslice, zslice)

        if ierr>0:
            raise ValueError('c_hydrodiy_gis.slice returns '+str(ierr))

        return zslice


    def plot(self, ax, *args, **kwargs):
        ''' Plot the grid using imshow. This is a basic plotting
        function. For more advanced plots, use
        hydrodiy.plot.gridplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the grid on

        '''

        xll, yll, csz, nr, nc = self._getsize()
        extent = [xll, xll+csz*nc, yll, yll+csz*nr]
        cax = ax.imshow(self.data, extent=extent, *args, **kwargs)
        return cax


    def clone(self, dtype=None):
        ''' Clone the current grid object and change dtype if needed

        Parameters
        -----------
        dtype : numpy.dtype
            Variable type of the cloned grid

        Returns
        -----------
        clone : hydrodiy.gis.grid.Grid
            Cloned grid
        '''

        grid = copy.deepcopy(self)

        if not dtype is None:
            grid.dtype = dtype

        return grid


    def clip(self, xll, yll, xur, yur):
        ''' Clip the current grid to a smaller area '''

        xy = [[xll, yll], [xur, yur]]
        idxcell0, idxcell1 = self.coord2cell(xy)

        ncols = self.ncols
        nx0 = idxcell0%ncols
        ny0 = (idxcell0-nx0)//ncols

        nx1 = idxcell1%ncols
        ny1 = (idxcell1-nx1)//ncols

        # Create grid
        name = self.name + '_clip'
        ncols = nx1+1-nx0
        nrows = ny0+1-ny1
        xy = self.cell2coord(idxcell0)
        xll = xy[0, 0]
        yll = xy[0, 1]

        grid = Grid(name, ncols, nrows, cellsize=self.cellsize,
                xllcorner=xll, yllcorner=yll,
                dtype=self.dtype,
                nodata_value=self.nodata_value)

        # Set data
        dt = self._data[ny1:ny0+1, nx0:nx1+1]
        grid.data = dt

        return grid


    def apply(self, fun, *args, **kwargs):
        ''' Apply a function to the grid data '''

        grid = self.clone()
        grid._data = fun(grid._data, *args, **kwargs).astype(self.dtype)

        return grid



class Catchment(object):
    ''' Catchment delineation tool '''

    def __init__(self, name, flowdir):
        self.name = name
        self._flowdir = flowdir.clone(np.int64)
        self._idxoutlet = None
        self._idxinlets = None
        self._idxcells_area = None
        self._idxcells_boundary = None
        self._xycells_boundary = None


    def __str__(self):
        str = '\nGrid {0}:\n'.format(self.name)
        str += '\toutlet   : {0}\n'.format(self._idxoutlet)
        str += '\tinlets   : {0}\n'.format(self._idxinlets)

        narea = 0
        if not self._idxcells_area is None:
            narea = len(self._idxcells_area)
        str += '\tarea     : {0} cells\n'.format(narea)

        nboundary = 0
        if not self._idxcells_boundary is None:
            nboundary = len(self._idxcells_boundary)
        str += '\tboundary : {0} cells\n'.format(nboundary)

        return str


    def __add__(self, other):
        ''' Combine two catchment areas '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, please' + \
                        ' delineate the area')

        if other._idxcells_area is None:
            raise ValueError('idxcells_area is None for other, please' + \
                        ' delineate the area')

        catchment = self.clone()
        catchment.name = self.name +'+'+other.name
        catchment._idxoutlet = None
        catchment._idxinlets = None
        catchment._idxcells_boundary = None
        catchment._xycells_boundary = None
        catchment._idxcells_area = np.union1d(self._idxcells_area,
                                    other._idxcells_area)
        return catchment


    def __sub__(self, other):
        ''' Substract two catchment areas '''
        catchment = self.clone()
        catchment.name = self.name +'-'+other.name
        catchment._idxoutlet = None
        catchment._idxinlets = None
        catchment._idxcells_boundary = None
        catchment._xycells_boundary = None

        catchment._idxcells_area = np.setdiff1d(self._idxcells_area,
                                    other._idxcells_area)

        return catchment


    @classmethod
    def from_dict(cls, dic):
        ''' Create catchment from dictionary

        Parameters
        -----------
        dic : dict
            Dictionary containing catchment attributes

        Returns
        -----------
        catchment : hydrodiy.grid.Catchment
            Catchment instance
        '''
        flowdir = Grid.from_dict(dic['flowdir'])
        catchment = Catchment(dic['name'], flowdir)
        catchment._idxoutlet = dic['idxoutlet']
        catchment._idxintlets = dic['idxinlets']

        catchment._idxcells_area = \
                        np.array(dic['idxcells_area']).astype(np.int64)

        return catchment


    @property
    def idxoutlet(self):
        ''' Get outlet cell '''
        return self._idxoutlet


    @property
    def idxinlets(self):
        ''' Get inlet cells '''
        return self._idxinlets


    @property
    def idxcells_area(self):
        ''' Get cells of catchment area '''
        return self._idxcells_area


    @property
    def xycells_boundary(self):
        ''' Get xy coords of catchment boundary '''
        return self._xycells_boundary


    @property
    def idxcells_boundary(self):
        ''' Get cells of catchment boundary '''
        return self._idxcells_boundary

    @property
    def flowdir(self):
        ''' Get flow direction grid '''
        return self._flowdir

    def clone(self):
        ''' Clone the current catchment '''
        catchment = copy.deepcopy(self)

        return catchment


    def extent(self):
        ''' Get catchment area extent '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, please' + \
                        ' delineate the area')

        xy = self._flowdir.cell2coord(self._idxcells_area)
        return np.min(xy[:, 0]), np.max(xy[:, 0]), \
                np.min(xy[:, 1]), np.max(xy[:, 1]),


    def upstream(self, idxdown):
        ''' Get upstream cell of a given cell '''
        idxdown = np.atleast_1d(idxdown).astype(np.int64)
        idxup = np.zeros((len(idxdown), 9), dtype=np.int64)
        ierr = c_hydrodiy_gis.upstream(FLOWDIRCODE,
                    self._flowdir.data, idxdown, idxup)

        if ierr>0:
            raise ValueError('c_hydrodiy_gis.upstream' + \
                                ' returns '+str(ierr))

        return idxup


    def downstream(self, idxup):
        ''' Get downstream cell of a given cell '''
        idxup = np.atleast_1d(idxup).astype(np.int64)
        idxdown = np.zeros(len(idxup), dtype=np.int64)

        ierr = c_hydrodiy_gis.downstream(FLOWDIRCODE,
                    self._flowdir.data, idxup, idxdown)

        if ierr>0:
            raise ValueError('c_hydrodiy_gis.downstream' + \
                                ' returns '+str(ierr))

        return idxdown


    def delineate_area(self, idxoutlet, idxinlets=None, nval=1000000):
        ''' Delineate catchment area from flow direction grid

        Parameters
        -----------
        idxoutlet : int
            Index of outlet cell
        idxinlets : list
            Index of inlet cells
        nval : int
            Maximum number of cells in area
        '''
        self._idxoutlet = np.int64(idxoutlet)

        if idxinlets is None:
            idxinlets = -1*np.ones(0, dtype=np.int64)
        else:
            idxinlets = np.atleast_1d(idxinlets).astype(np.int64)
            self._idxinlets = idxinlets

        idxcells = -1*np.ones(nval, dtype=np.int64)
        buffer1 = -1*np.ones(nval, dtype=np.int64)
        buffer2 = -1*np.ones(nval, dtype=np.int64)

        # Compute area
        ierr = c_hydrodiy_gis.delineate_area(FLOWDIRCODE,
                    self._flowdir.data, self._idxoutlet, idxinlets,
                    idxcells, buffer1, buffer2)

        if ierr>0:
            raise ValueError(('c_hydrodiy_gis.delineate_area' + \
                ' returns {0}. Consider increasing ' + \
                'buffer size ({1})').format(ierr, nval))

        idx = idxcells >= 0
        self._idxcells_area = idxcells[idx]


    def delineate_boundary(self):
        ''' Delineate catchment boundary from area '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, please' + \
                        ' delineate the area')

        nrows = self._flowdir.nrows
        ncols = self._flowdir.ncols

        # Initialise boundary cells with vector of same length
        # than area
        nval = np.int64(len(self._idxcells_area))
        idxcells_boundary = -1*np.ones(nval, dtype=np.int64)
        buf = -1*np.ones(nval, dtype=np.int64)
        grid = np.zeros(nrows*ncols, dtype=np.int64)
        grid[self._idxcells_area] = 1

        # Compute boundary with varying dmax
        # This point could be improved!!
        ierr = c_hydrodiy_gis.delineate_boundary(nrows, ncols, \
                    self._idxcells_area, \
                    buf, grid, idxcells_boundary)

        if ierr>0:
            raise ValueError(('c_hydrodiy_gis.delineate_boundary' +
                ' returns {0}').format(ierr))

        idx = idxcells_boundary >= 0
        idxcells_boundary = idxcells_boundary[idx]
        self._idxcells_boundary = idxcells_boundary

        # Generate coordinates for the boundary
        self._xycells_boundary = self._flowdir.cell2coord(
                                        self._idxcells_boundary)


    def intersect(self, grid):
        ''' Intersect catchment area with other grid '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, ' + \
                'please delineate the area')

        if len(self._idxcells_area)==0:
            raise ValueError('No cells in idxcells_area, ' + \
                'please use another outlet')

        xll, yll, csz, nrows, ncols = grid._getsize()
        _, _, csz_area, _, _ = self._flowdir._getsize()

        xy_area = self._flowdir.cell2coord(self._idxcells_area)
        npoints = np.zeros((1,), dtype=np.int64)
        idxcells = np.zeros(nrows*ncols, dtype=np.int64)
        weights = np.zeros(nrows*ncols, dtype=np.float64)

        # Compute river path
        ierr = c_hydrodiy_gis.intersect(nrows, ncols,
            xll, yll, csz, csz_area,
            xy_area, npoints, idxcells, weights)

        if ierr>0:
            raise ValueError(('c_hydrodiy_gis.intersect' +
                ' returns {0}').format(ierr))

        idxcells = idxcells[:npoints[0]]
        weights = weights[:npoints[0]]

        return idxcells, weights


    def isin(self, idxcell):
        ''' Check if a cell is within the catchment area '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, ' + \
                'please delineate the area')

        return idxcell in self._idxcells_area


    def compute_area(self, to_proj, from_proj=None):
        ''' Compute catchment area in km2 using a projection and
            applying the Shoelace algorithm.
            See https://en.wikipedia.org/wiki/Shoelace_formula

        Parameters
        -----------
        to_proj : pyproj.Proj
            Target projection obtained from the pyproj package.
            For example for the GDA94, we have
            to_proj = pyproj.Proj('+init=EPSG:3112')

        from_proj : pyproj.Proj
            Projection of grid coordinates obtained from the pyproj package.
            If None, it is assumed that the original coordinates are not
            projected.

            For example for the GDa94, we have
            from_proj = pyproj.Proj('+init=EPSG:4326')

        Returns
        -----------
        area : float
            Catchment area in km2
        '''
        idxb = self.idxcells_boundary
        if idxb is None:
            raise ValueError('idxcells_boundary is None, ' + \
                'please delineate the area')

        # Obtain source coordinates in source projection system
        xy = self.flowdir.cell2coord(idxb)
        if from_proj is None:
            xy1 = xy
        else:
            xy1 = np.array([from_proj(uv[0], uv[1]) for uv in xy])

        # Project to targe projection system
        xy2 = np.array([to_proj(uv[0], uv[1]) for uv in xy1])

        # Compute area with Shoelace formula
        x = xy2[:, 0]
        y = xy2[:, 1]
        area = 0.5e-6*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

        return area



    def plot_area(self, ax, *args, **kwargs):
        ''' Plot catchment area '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_boundary is None, ' + \
                'please delineate the area')

        xy = self._flowdir.cell2coord(self._idxcells_area)
        ax.plot(xy[:, 0], xy[:, 1], *args, **kwargs)


    def plot_boundary(self, ax, *args, **kwargs):
        ''' Plot catchment boundary '''

        if self._idxcells_boundary is None:
            raise ValueError('idxcells_boundary is None, ' + \
                'please delineate the area')

        xy = self._flowdir.cell2coord(self._idxcells_boundary)
        ax.plot(xy[:, 0], xy[:, 1], *args, **kwargs)


    def to_dict(self):
        ''' Export data to a dictionnary '''

        if self._idxcells_area is None:
            raise ValueError('idxcells_area is None, ' + \
                'please delineate the area')

        inlets = None
        if not self._idxinlets is None:
            inlets = list(self._idxinlets)

        dic = {
            'name': self.name,
            'idxoutlet':self._idxoutlet,
            'idxinlets':inlets,
            'idxcells_area':list(self._idxcells_area),
            'flowdir':self._flowdir.to_dict(),
        }
        return dic


    def load(self, filename):
        ''' Load data from a JSON file '''

        if not filename.endswith('json'):
            raise ValueError(('Filename ({0}) should end with a' + \
                ' bil extension').format(filename))



def delineate_river(flowdir, idxupstream, nval=1000000):
    ''' Delineate river upstream point and flow direction grid

    Parameters
    -----------
    flowdir : hydrodiy.grid.Grid
        Flow direction grid
    idxuptstream : int
        Index of upstream cell
    nval : int
        Number of cells to go downstream (length of river)

    Returns
    -----------
    data : pandas.DataFrame
        Arrays containing river data:
        - Col1 = Distance from upstream cell (in cell numbers)
        - Col2 = Displacement in x axis (in cell number)
        - Col3 = Displacement in y axis (in cell number)
        - Col4 = x coordinate of point
        - Col5 = y coordinate of point
        - Col6 = Index of cells forming the river
    '''
    # Check type of flowdir
    flowdir.dtype = np.int64

    # Get data from flowdir
    xll, yll, csz, nr, nc = flowdir._getsize()

    # Allocate memory
    idxupstream = np.int64(idxupstream)
    idxcells = -1*np.ones(nval, dtype=np.int64)
    data = np.zeros((nval, 5), dtype=np.float64)
    npoints = np.zeros((1,), dtype=np.int64)

    # Compute river path
    ierr = c_hydrodiy_gis.delineate_river(xll, yll, csz,
                FLOWDIRCODE, flowdir.data,
                idxupstream, npoints, idxcells, data)

    if ierr>0:
        raise ValueError(('c_hydrodiy_gis.delineate_river' +
            ' returns {0}').format(ierr))

    nval = npoints[0]
    data = pd.DataFrame(data[:nval],
        columns=['dist', 'dx', 'dy', 'x', 'y'])
    data['idxcell'] = idxcells[:nval]

    return data


def accumulate(flowdir, nprint=100, maxarea=-1):
    ''' Compute flow accumulation from the flow direction grid '''

    nprint = np.int64(nprint)

    if maxarea == -1:
        maxarea = flowdir.nrows * flowdir.ncols
    maxarea = np.int64(maxarea)

    # Convert flowdir
    flowdir.dtype = np.int64

    # Initiase the accumulation grid with 0 accumulation
    accumulation = flowdir.clone()
    accumulation.fill(1)

    # Compute accumulation
    ierr = c_hydrodiy_gis.accumulate(nprint, maxarea, FLOWDIRCODE,
                flowdir.data, accumulation.data)

    if ierr>0:
        raise ValueError(('c_hydrodiy_gis.accumulate' +
            ' returns {0}').format(ierr))

    return accumulation


def voronoi(catchment, xypoints):
    ''' Compute the Voronoi weights for a set of points close to a catchment

    Parameters
    -----------
    catchment : hydrodiy.gis.grid.Catchment
        Catchment of interest. Catchment area must be delineated (via
        catchment.delineate_area)
    xypoints : numpy.ndarray
        X and Y coordinates of points

    Returns
    -----------
    weights : numpy.ndarray
        Weights for each of the points as per a Voronoi diagram
        (i.e. percentage of Voronoi cell falling the catchment area
        for each point)
    '''

    if catchment._idxcells_area is None:
        raise ValueError('Catchment idxcells_area is None, ' + \
            'please delineate the area')

    xll, yll, csz, nrows, ncols = catchment._flowdir._getsize()

    idxcells_area = np.array(catchment._idxcells_area).astype(np.int64)
    xypoints = np.atleast_2d(xypoints).astype(np.float64)
    weights = np.zeros(xypoints.shape[0]).astype(np.float64)

    ierr = c_hydrodiy_gis.voronoi(nrows, ncols, xll, yll, csz,
                idxcells_area, xypoints, weights)

    if ierr>0:
        raise ValueError(('c_hydrodiy_gis.voronoi' +
            ' returns {0}').format(ierr))

    return weights



def get_mask(name):
    ''' Get reference gridss defined in
    Bureau of meteorology products

    Parameters
    -----------
    name : str
        Name of grid :
        - AWRAL : Grid used in AWRAL products
        - AWAP : Grid used in AWAP products

    Returns
    -----------
    gr : hydrodiy.gis.grid.Grid
        Mask grid containing 1 for cells within the grid
        and 0 for cells outside the grid

    '''

    expected = ['AWRAL', 'AWAP', 'WATERDYN']
    if not name in expected:
        raise ValueError('Expected name in {0}, got {1}'.format(
            '/'.join(expected), name))

    fbase = '{0}_GRID'.format(name)
    fzip = os.path.join(F_HYGIS_DATA, '{0}.zip'.format(fbase))
    fdata = os.path.join(F_HYGIS_DATA, '{0}.bil'.format(fbase))

    # Extract data from zipfile if it does not exist
    if not os.path.exists(fdata):
        with zipfile.ZipFile(fzip, 'r') as zipf:
            zipf.extract('{0}.bil'.format(fbase), F_HYGIS_DATA)
            zipf.extract('{0}.hdr'.format(fbase), F_HYGIS_DATA)

    # Reads data
    gr = Grid.from_header(fdata)

    return gr
