""" Module containing grid utilities """

import re
import os
from pathlib import Path
import copy
import zipfile
from io import StringIO
from tempfile import TemporaryFile
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, maximum_filter, \
                                binary_fill_holes

from hydrodiy.gis import gutils
from hydrodiy.io import csv

from hydrodiy import has_c_module
if has_c_module("gis", False):
    import c_hydrodiy_gis

# Codes indicating the flow direction for a cell
# using ESRI convention
# i.e.
#  32  64   128       \   |   /
#  16   0     1   =>  <-  x   ->
#   8   4     2       /   |   \
#
FLOWDIRCODE = np.array([[32, 64, 128],
                        [16, 0, 1],
                        [8, 4, 2]]).astype(np.int64)

# Path to hygis data
FHERE = Path(__file__).resolve().parent
F_HYGIS_DATA = FHERE / "data"

# AWRAL subgrids
FZIP_AWRAL_SUBGRIDS = F_HYGIS_DATA/"AWRAL_SUBGRIDS.zip"
with zipfile.ZipFile(str(FZIP_AWRAL_SUBGRIDS), "r") as archive:
    AWRAL_SUBGRIDS, _ = csv.read_csv("awra_grids.csv", archive=archive)

AWRAL_SUBGRIDS.loc[:, "gridid"] = "AWRAL_"\
                                  + AWRAL_SUBGRIDS.entity_type.str.upper() \
                                  + "_" + AWRAL_SUBGRIDS.name_short.str.upper()


class Grid(object):
    """ Gridded data object """

    def __init__(self, name, ncols, nrows=None, cellsize=1.,
                 xllcorner=0, yllcorner=0, dtype=np.float64,
                 nodata=0,
                 comment=""):

        if nrows is None:
            nrows = ncols

        self.name = name
        self.ncols = np.int64(ncols)
        self.nrows = np.int64(nrows)
        self._dtype = dtype
        self._nodata = nodata

        self.cellsize = np.float64(cellsize)

        self.xllcorner = np.float64(xllcorner)
        self.yllcorner = np.float64(yllcorner)

        self.nodata = nodata
        self._mindata = -np.inf
        self._maxdata = np.inf
        self.comment = comment

        self._data = np.zeros((self.nrows, self.ncols), dtype=dtype)

    def _getsize(self):
        """ Returns dimensions of the grid """
        xll = self.xllcorner
        yll = self.yllcorner
        csz = self.cellsize
        nrows = self.nrows
        ncols = self.ncols
        return xll, yll, csz, nrows, ncols

    def __setitem__(self, index, value):
        """ Set cell values """
        index = np.int64(index)
        self._data.flat[index] = np.array(value).astype(self.dtype)

    def __getitem__(self, index):
        """ Extract cell values """
        index = np.int64(index)
        return self._data.flat[index]

    def __str__(self):
        str = "\nGrid {0}:\n".format(self.name)
        str += "\tncols    : {0}\n".format(self.ncols)
        str += "\tnrows    : {0}\n".format(self.nrows)
        str += "\tcellsize : {0}\n".format(self.cellsize)
        str += "\txllcorner: {0}\n".format(self.xllcorner)
        str += "\tyllcorner: {0}\n".format(self.yllcorner)
        str += "\tdtype    : {0}\n".format(self.dtype)
        str += "\tno_data_value : {0}\n".format(self.nodata)
        str += "\tcomment  : {0}\n".format(self.comment)

        # Add attributes from parent grid if any
        if hasattr(self, "parentgrid_ncols"):
            str += "\n\t--- PARENT GRID ---\n"

        for attr in ["name", "ncols", "nrows", "cellsize",
                     "xllcorner", "yllcorner", "rows_start",
                     "rows_end", "cols_start", "cols_end"]:
            pattr = "parentgrid_"+attr
            if hasattr(self, pattr):
                str += "\t{} : {}\n".format(re.sub("parentgrid_", "", pattr),
                                            getattr(self, pattr))
        return str

    @classmethod
    def from_stream(cls, stream_header, stream_data=None):
        """ Create grid from file-like stream.

        Parameters
        -----------
        stream_header : io.StringIO or _io.TextIOWrapper
            File-like stream pointing towards header file.

        stream_data : io.BytesIO or _io.BufferedReader
            File-like stream pointing towards data file.

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        """
        # initialise config
        config = {
            "xllcorner": 0.,
            "yllcorner": 0.,
            "cellsize": 1.,
            "nodata": 0,
            "nbits": 64,
            "pixeltype": "float",
            "byteorder": "i",
            "comment": "No comment"
        }

        # Define name if available
        if hasattr(stream_header, "name"):
            basename = os.path.basename(stream_header.name)
            config["name"] = os.path.splitext(basename)[0]
        else:
            config["name"] = "no_name"

        parent_config = {}

        # Read property from header
        stream_header.seek(0)
        for line in stream_header.readlines():
            line = re.split(" ", re.sub(" +", " ", line))
            pname = line[0].lower()
            try:
                if pname in ["pixeltype", "byteorder",
                             "layout", "comment", "name"]:
                    pvalue = " ".join(line[1:]).strip().lower()
                elif pname.startswith("n") and not pname.startswith("nodata"):
                    pvalue = int(line[1].strip())
                elif pname.startswith("parentgrid_n"):
                    pvalue = int(line[1].strip())
                else:
                    pvalue = float(line[1].strip())

                if pname.startswith("parent"):
                    parent_config[pname] = pvalue
                else:
                    config[pname] = pvalue

            except ValueError:
                errmess = f"Header field {pname} cannot be processed."
                warnings.warn(errmess)
                pass

        # Define dtype
        if config["byteorder"] == "m":
            byteorder = ">"
        elif config["byteorder"] == "i":
            byteorder = "<"
        else:
            errmess = "Byteorder {config['byteorder']} not recognised"
            raise ValueError(errmess)

        pixeltype = re.sub("nsignedint$|^signed|nt|loat", "",
                           config["pixeltype"])
        nbits = config["nbits"]//8
        config["dtype"] = np.dtype(byteorder +
                                   pixeltype + str(nbits)).type

        # Check cell size / dimensions
        if "xdim" in config:
            config["cellsize"] = config["xdim"]

            # Check the cell are squared (i.e. xdim=ydim)
            if "ydim" in config:
                if abs(config["ydim"] - config["xdim"]) > 1e-10:
                    errmess = f"xdim {config['xdim']} != "\
                              + f"ydim {config['ydim']}."
                    raise ValueError(errmess)

        if "ulxmap" in config:
            config["xllcorner"] = config["ulxmap"]
            yll = config["ulymap"] - config["cellsize"]*config["nrows"]
            config["yllcorner"] = yll

        # Rename nodata_value to nodata
        if "nodata_value" in config:
            config["nodata"] = config["nodata_value"]

        # Filters config data
        keys = ["name", "ncols", "nrows", "cellsize", "comment",
                "xllcorner", "yllcorner", "dtype", "nodata"]
        config = {k: config[k] for k in config if k in keys}

        # Creates grid
        grid = Grid(**config)

        # Reads data if bil file is there
        if stream_data is not None:
            stream_data.seek(0)
            grid.load(stream_data)

        # Adds parent meta data
        if len(parent_config) > 0:
            for key, value in parent_config.items():
                setattr(grid, key, value)

        return grid

    @classmethod
    def from_header(cls, fileheader):
        """ Create grid from header file

        Parameters
        -----------
        fileheader : str
            Path to the header file (or BIL file)

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        """
        fileheader = Path(fileheader)

        # Generate file paths
        fileheader = fileheader.parent / (fileheader.stem+".hdr")
        filedata = fileheader.parent / (fileheader.stem+".bil")

        if not fileheader.exists():
            errmess = f"File {fileheader} does not exist."
            raise ValueError(errmess)

        with fileheader.open("r") as fh:
            if filedata.exists():
                with filedata.open("rb") as fd:
                    return cls.from_stream(fh, fd)
            else:
                return cls.from_stream(fh)

    @classmethod
    def from_zip(cls, zipfilepath, fileheader_in_zip):
        """ Create grid from zip file

        Parameters
        -----------
        zipfilepath : str
            Path to the zip file
        fileheader_in_zip : str
            Path to the header file within the zip archive

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        """
        # Generate file paths
        base = os.path.splitext(fileheader_in_zip)[0]
        fileheader = base + ".hdr"
        filedata = base + ".bil"

        # Open zipfile
        with zipfile.ZipFile(zipfilepath, "r") as archive:
            fh = StringIO(archive.open(fileheader).read().decode())

            if filedata not in archive.namelist():
                # No data, just header
                return Grid.from_stream(fh)
            else:
                with TemporaryFile() as fd:
                    # Reads data to temporary file
                    # (numpy does not support reading from a BytesIO stream)
                    fd.write(archive.open(filedata).read())

                    return Grid.from_stream(fh, fd)

    @classmethod
    def from_dict(cls, dic):
        """ Create grid from dictionary

        Parameters
        -----------
        dic : dict
            Dictionary

        Returns
        -----------
        grid : hydrodiy.grid.Grid
            Grid instance
        """

        # Init argument
        dic2 = {"name": dic["name"], "ncols": dic["ncols"]}

        # Init optional arguments
        for opt in ["nrows", "cellsize", "xllcorner", "yllcorner",
                    "dtype", "nodata", "comment"]:
            if opt in dic:
                dic2[opt] = dic[opt]

        if "dtype" in dic2:
            dic2["dtype"] = np.dtype(dic2["dtype"]).type

        return Grid(**dic2)

    @property
    def shape(self):
        """ Return data grid shape """
        return self._data.shape

    @property
    def xlim(self):
        """ Return x limits """
        return self.xllcorner, self.xllcorner+self.ncols*self.cellsize

    @property
    def ylim(self):
        """ Return y limits """
        return self.yllcorner, self.yllcorner+self.nrows*self.cellsize

    @property
    def xvalues(self):
        """ Return coordinates along grid columns (x) """
        cells = np.arange(self.ncols)
        xv = self.cell2coord(cells)
        return xv[:, 0]

    @property
    def yvalues(self):
        """ Return coordinates along grid rows (y) """
        cells = np.arange(0, self.nrows*self.ncols, self.ncols)
        xv = self.cell2coord(cells)
        return xv[:, 1]

    @property
    def data(self):
        """ Get grid data """
        return self._data

    @data.setter
    def data(self, value):
        """ Set grid data """

        _value = np.ascontiguousarray(np.atleast_2d(value))

        if _value.ndim != 2:
            raise ValueError("Expected 2d array, got "
                             + f"{_value.ndim} dimensions.")

        nrows, ncols = _value.shape

        if nrows != self.nrows:
            errmess = "Wrong number of rows:"\
                      + f" data has {nrows}, but expected {self.nrows}."
            raise ValueError(errmess)

        if ncols != self.ncols:
            errmess = "Wrong number of columns:"\
                      + f" data has {ncols}, but expects {self.ncols}."
            raise ValueError(errmess)

        self._data = np.clip(_value, self.mindata,
                             self.maxdata).astype(self.dtype)

    @property
    def nodata(self):
        """ Get nodata value """
        return self._nodata

    @nodata.setter
    def nodata(self, value):
        """ Set nodata value """
        self._nodata = self.dtype(value)

    @property
    def dtype(self):
        """ Get data type """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """ Set data type """
        self._dtype = value
        self._data = self._data.astype(value)

    @property
    def mindata(self):
        """ Get data minimum allowed """
        return self._mindata

    @mindata.setter
    def mindata(self, value):
        """ Set data minimum allowed """
        self._mindata = self.dtype(value)
        if self._mindata > self._maxdata:
            errmess = "Expected mindata<maxdata,"\
                      + f" got mindata={self._mindata}"\
                      + f" and maxdata={self._maxdata}."
            raise ValueError(errmess)

        self._data = np.maximum(self._data, self._mindata)

    @property
    def maxdata(self):
        """ Get data maximum allowed """
        return self._maxdata

    @maxdata.setter
    def maxdata(self, value):
        """ Set data maximum allowed """
        self._maxdata = self.dtype(value)
        if self._mindata > self._maxdata:
            errmess = "Expected mindata<maxdata,"\
                      + f" got mindata={self._mindata}"\
                      + f" and maxdata={self._maxdata}."
            raise ValueError(errmess)

        self._data = np.minimum(self._data, self._maxdata)

    def set_parent_attributes(self, grid, row_start, row_end,
                              col_start, col_end):
        """ Set parent attributes when clipping a grid """

        for attr in ["name", "ncols", "nrows", "cellsize",
                     "xllcorner", "yllcorner"]:
            setattr(self, "parentgrid_"+attr, getattr(grid, attr))

        self.parentgrid_rows_start = row_start
        self.parentgrid_rows_end = row_end
        self.parentgrid_cols_start = col_start
        self.parentgrid_cols_end = col_end

    def same_geometry(self, grd):
        """ Check another grid has the same geometry

        Parameters
        -----------
        grd : hydrodiy.gis.grid.Grid
            External grid

        Returns
        -----------
        identical : bool
            Same geometry True/False
        """
        identical = grd.ncols == self.ncols
        identical = identical & (grd.nrows == self.nrows)
        identical = identical & np.isclose(grd.xllcorner, self.xllcorner)
        identical = identical & np.isclose(grd.yllcorner, self.yllcorner)
        identical = identical & np.isclose(grd.cellsize, self.cellsize)

        return identical

    def load(self, stream_data):
        """ Load data from file

        Parameters
        -----------
        stream_data : io.ByteIO or str
            Stream to binary data (only BIL file format at the moment) or
            File path.
        """
        data = np.fromfile(stream_data, self.dtype)

        nval = self.nrows * self.ncols
        if len(data) != nval:
            errmess = f"File contains {len(data)} data points,"\
                      + f" expecting {nval}."
            raise ValueError(errmess)

        self._data = np.clip(data.reshape((self.nrows, self.ncols)),
                             self.mindata, self.maxdata).astype(self.dtype)

    def to_dict(self):
        """ Export grid metadata to json """
        js = {
            "name": self.name,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "cellsize": self.cellsize,
            "xllcorner": self.xllcorner,
            "yllcorner": self.yllcorner,
            "dtype": np.dtype(self.dtype).str,
            "nodata": str(self.nodata),
            "comment": self.comment
            }

        # Add attributes from parent grid if any
        for attr in ["nrows", "ncols", "xllcorner",
                     "yllcorner", "rows_start", "rows_end",
                     "cols_start", "cols_end"]:
            pattr = "parentgrid_"+attr
            if hasattr(self, pattr):
                js[pattr] = getattr(self, pattr)

        return js

    def save(self, filename):
        """ Save data as a BIL file """
        filename = str(filename)

        if not filename.endswith("bil"):
            errmess = f"Filename {filename}' should end with"\
                      + " a bil extension."
            raise ValueError(errmess)

        # Print header file
        fileheader = re.sub("bil$", "hdr", filename)
        with open(fileheader, "w") as fh:
            for attname in ["nrows", "ncols", "xllcorner",
                            "yllcorner", "cellsize"]:
                attval = getattr(self, attname)
                fh.write("{0:<14} {1}\n".format(attname.upper(), attval))

            # nbits
            ddtype = np.dtype(self.dtype)
            nbits = ddtype.itemsize*8
            fh.write("{0:<14} {1}\n".format("NBITS", nbits))

            # pixel type
            name = re.sub("[0-9]+$", "", ddtype.name)
            if name == "int":
                pixeltype = "signedint"
            elif name == "uint":
                pixeltype = "unsignedint"
            elif name == "float":
                pixeltype = "float"
            else:
                raise ValueError("Type name {0} unrecognised".format(name))

            fh.write("{0:<14} {1}\n".format("PIXELTYPE", pixeltype.upper()))

            # Byte order
            if ddtype.byteorder == ">":
                byteorder = "M"
            else:
                byteorder = "I"
            fh.write("{0:<14} {1}\n".format("BYTEORDER", byteorder))

            # Name
            fh.write("{0:<14} {1}\n".format("NAME", self.name))

            # Comment
            comment = self.comment
            if comment == "":
                comment = "No comment"
            fh.write("{0:<14} {1}\n".format("COMMENT", comment))

            # Parent attributes
            for attr in ["nrows", "ncols", "xllcorner",
                         "yllcorner", "rows_start", "rows_end",
                         "cols_start", "cols_end"]:
                pattr = "parentgrid_"+attr
                if hasattr(self, pattr):
                    fh.write("{0:<22} {1}\n".format(pattr.upper(),
                                                    getattr(self, pattr)))

        # Print data
        self._data.tofile(filename)

    def fill(self, value):
        """ Initialise grid value

        Parameters
        -----------
        value : float
            Value used to fill the grid data.

        """
        dtype = self._dtype
        self.data.fill(dtype(value))

    def coord2cell(self, xycoords):
        """ Return cell number from coordinates.
        Cells are counted from the top left corner,
        row by row from top to bottom. Example for a 4x4 grid:
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

        """
        has_c_module("gis")

        xll, yll, csz, nrows, ncols = self._getsize()

        xycoords = np.ascontiguousarray(np.atleast_2d(xycoords),
                                        dtype=np.float64)
        idxcell = np.zeros(len(xycoords)).astype(np.int64)

        ierr = c_hydrodiy_gis.coord2cell(nrows, ncols, xll, yll,
                                         csz, xycoords, idxcell)
        if ierr > 0:
            errmess = f"c_hydrodiy_gis.coord2cell returns {ierr}."
            raise ValueError(errmess)

        return idxcell

    def cell2coord(self, idxcells):
        """ Return coordinate from cell number.
        For idxcells data, cells are counted from the top left corner,
        row by row from top to bottom.

        For coords data, x coords increase from left to right and
        y coords increase from bottom to top.

        Example for a 4x4 grid with (xll,yll)=(0,0) and cellsize=1:
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

        """
        has_c_module("gis")

        xll, yll, csz, nrows, ncols = self._getsize()
        idxcells = np.ascontiguousarray(np.atleast_1d(idxcells),
                                        dtype=np.int64)
        xycoords = np.zeros((len(idxcells), 2)).astype(np.float64)

        ierr = c_hydrodiy_gis.cell2coord(nrows, ncols, xll, yll, csz,
                                         idxcells, xycoords)
        if ierr > 0:
            errmess = f"c_hydrodiy_gis.cell2coord returns {ierr}."
            raise ValueError(errmess)

        return xycoords

    def cell2rowcol(self, idxcells):
        """ Return row and column number from cell number
        For idxcells data, cells are counted from the top left corner,
        row by row from top to bottom.

        For rowcol data, columns are counted from left to right.
        rows are counted from top to bottom.

        Example for a 4x4 grid with (xll,yll)=(0,0):
         0  1  2  3        (0, 0)  (0, 1) (0, 2) (0, 3)
         4  5  6  7   ->   (1, 0)  (1, 1) (1, 2) (1, 3)
         8  9 10 11        (2, 0)  (2, 1) (2, 2) (2, 3)
        12 13 14 15        (3, 0)  (3, 1) (3, 2) (3, 3)

        Parameters
        -----------
        idxcell : numpy.ndarray
            1D array containing cell numbers

        Returns
        -----------
        rowcols : numpy.ndarray
            2D array with row first column and
            column in second column.

        """
        has_c_module("gis")
        xll, yll, csz, nrows, ncols = self._getsize()

        idxcells = np.ascontiguousarray(np.atleast_1d(idxcells),
                                        dtype=np.int64)
        rowcols = np.zeros((len(idxcells), 2)).astype(np.int64)

        ierr = c_hydrodiy_gis.cell2rowcol(nrows, ncols,
                                          idxcells, rowcols)
        if ierr > 0:
            errmess = "c_hydrodiy_gir.cell2rowcol returns {ierr}."
            raise ValueError(errmess)

        return rowcols

    def neighbours(self, idxcell):
        """ Compute the codes of cells surrounding a cell.
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

        """
        has_c_module("gis")

        _, _, _, nrows, ncols = self._getsize()

        idxcell = np.int64(idxcell)
        neighbours = np.zeros(9).astype(np.int64)

        ierr = c_hydrodiy_gis.neighbours(nrows, ncols, idxcell,
                                         neighbours)
        if ierr > 0:
            errmess = f"c_hydrodiy_gis.neighbours returns {ierr}."
            raise ValueError(errmess)

        return neighbours

    def slice(self, xyslice):
        """ Extract a profile from the grid

        Parameters
        -----------
        xyslice : numpy.ndarray
            2D array with x coords in first column and
            y coords in second column.

        Returns
        -----------
        zslice : numpy.ndarray
            1D array containing sliced values from gridded data
        """
        has_c_module("gis")

        # Get inputs
        xll, yll, csz, _, _ = self._getsize()

        xyslice = np.ascontiguousarray(np.atleast_2d(xyslice),
                                       dtype=np.float64)
        zslice = np.zeros(len(xyslice)).astype(np.float64)

        # Run C code
        ierr = c_hydrodiy_gis.slice(xll, yll, csz,
                                    self._data.astype(np.float64),
                                    xyslice, zslice)

        if ierr > 0:
            raise ValueError("c_hydrodiy_gis.slice returns "+str(ierr))

        return zslice

    def plot(self, ax, *args, **kwargs):
        """ Plot the grid using imshow. This is a basic plotting
        function. For more advanced plots, use
        hydrodiy.plot.gridplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the grid on

        args, kwargs: arguments passed to ax.imshow
        """
        xll, yll, csz, nr, nc = self._getsize()
        extent = [xll, xll+csz*nc, yll, yll+csz*nr]
        if np.any(np.isnan(extent)):
            errmess = "Cannot plot grid when one of"\
                      + f" xllcorner ({xll}), yllcorner ({yll})"\
                      + f" cellsize ({csz}), nrows ({nr})"\
                      + f" or ncols ({nc}) is nan"
            raise ValueError(errmess)

        return ax.imshow(self.data, extent=extent, *args, **kwargs)

    def plot_values(self, ax, fmt="0.2f", mini=-np.inf,
                    maxi=np.inf, *args, **kwargs):
        """ Plot grid values. This is a basic plotting
        function. For more advanced plots, use
        hydrodiy.plot.gridplot

        Parameters
        -----------
        ax : matplotlib.axes
            Axe to draw the grid on
        fmt : str
            Number formatting
        mini : float
            Minimum value to plot.
        maxi : float
            Maximum value to plot

        args, kwargs: arguments passed to ax.text

        Returns
        -----------
        txt : list
            List of matplotlib text objects.
        """
        txt = []
        idxcells = np.arange(self.nrows*self.ncols)
        coords = self.cell2coord(idxcells)
        data = self.data.flat
        for i in idxcells:
            value = data[i]
            if value >= mini and value <= maxi:
                lab = "{:{fmt}}".format(value, fmt=fmt)
                t = ax.text(coords[i, 0], coords[i, 1], lab, va="center",
                            ha="center", *args, **kwargs)
                txt.append(t)

        return txt

    def clone(self, dtype=None):
        """ Clone the current grid object and change dtype if needed

        Parameters
        -----------
        dtype : numpy.dtype
            Variable type of the cloned grid

        Returns
        -----------
        clone : hydrodiy.gis.grid.Grid
            Cloned grid
        """
        grid = copy.deepcopy(self)
        if dtype is not None:
            grid.dtype = dtype

        return grid

    def clip(self, xll, yll, xur, yur):
        """ Clip the current grid to a smaller area """
        # Process coordinates
        xy = [[xll, yll], [xur, yur]]
        idxcell0, idxcell1 = self.coord2cell(xy)
        rowcols = self.cell2rowcol([idxcell0, idxcell1])

        # Create grid
        name = self.name + "_clip"
        nrows = rowcols[0, 0] - rowcols[1, 0]+1
        ncols = rowcols[1, 1] - rowcols[0, 1]+1
        xy = self.cell2coord(idxcell0)
        xllg = xy[0, 0]-self.cellsize/2
        yllg = xy[0, 1]-self.cellsize/2
        grid = Grid(name, ncols, nrows, cellsize=self.cellsize,
                    xllcorner=xllg, yllcorner=yllg,
                    dtype=self.dtype,
                    nodata=self.nodata)

        grid.comment = f"Clip of grid {self.name} on the"\
                       + f" box [{xll}, {yll}, {xur}, {yur}]."
        # Set data
        row0, row1 = rowcols[::-1][:, 0]
        col0, col1 = rowcols[:, 1]
        grid.data = self._data[row0:row1+1, col0:col1+1]

        # Set parent grid attributes
        grid.set_parent_attributes(self, row0, row1, col0, col1)

        return grid

    def apply(self, fun, *args, **kwargs):
        """ Apply a function to the grid data """
        grid = self.clone()
        grid._data = fun(grid._data, *args, **kwargs).astype(self.dtype)

        return grid

    def interpolate(self, grid, method="linear"):
        """ Interpolate the current grid based on geometry supplied
        by another grid. The interpolation is based on the scipy
        procedures interpolate.griddata

        Parameters
        -----------
        grid : hydrodiy.gis.grid.Grid
            Input grid used to define interpolation geometry
        method : str
            Interpolation method. See scipy.interpolate.griddata

        Returns
        -----------
        interp_grid : hydrodiy.gis.grid.Grid
            Interpolated grid matching the input grid geometry
        """
        # Skip the interpolation process if geometry is same
        if self.same_geometry(grid):
            return self.clone()

        # Build coordinate matrices
        xll, yll, csz, nr, nc = self._getsize()
        u = np.linspace(xll, xll+csz*nc, nc)
        v = np.linspace(yll, yll+csz*nr, nr)
        x, y = np.meshgrid(u, v)

        # Matrices for input geometry
        xll, yll, csz, nr, nc = grid._getsize()
        u = np.linspace(xll, xll+csz*nc, nc)
        v = np.linspace(yll, yll+csz*nr, nr)
        xnew, ynew = np.meshgrid(u, v)

        # interpolation
        z = self.data.flat
        xy = np.column_stack([x.flat, y.flat])
        znew = griddata(xy, z, (xnew, ynew), method=method)

        # Create grid
        interp_grid = grid.clone()
        interp_grid.dtype = self.dtype
        interp_grid.data = znew

        return interp_grid

    def cells_inside_polygon(self, polygon, atol=1e-8):
        """ Identify grid cells which have their centroid falling
        into a defined polygon.

        Parameters
        -----------
        polygon : numpy.array polygon
            Coordinates of polygon vertices given as 2d numpy array [x,y]
        atol : float
            Tolerance factor for float number identity testing

        Returns
        -----------
        cells_inside : pandas.DataFrame
            Grid cells in this polygon identified by x, y and cell number
        """
        # Get cell coordinates
        ncells = np.arange(self.nrows*self.ncols)
        points = self.cell2coord(ncells)

        # Get list of cells inside polygon
        inside = gutils.points_inside_polygon(points, polygon)
        inside = inside.astype(bool)

        dd = {
            "x": points[inside, 0],
            "y": points[inside, 1],
            "cell": ncells[inside]
            }
        return pd.DataFrame(dd)


class Catchment(object):
    """ Catchment delineation tool """

    def __init__(self, name, flowdir):
        self.name = name
        self._flowdir = flowdir.clone(np.int64)
        self._idxcell_outlet = None
        self._idxinlets = None
        self._idxcells_area = None
        self._idxcells_area_filled = None
        self._idxcells_boundary = None
        self._flowpathlengths = None
        self._xycells_boundary = None

    def __str__(self):
        str = "\nGrid {0}:\n".format(self.name)
        str += "\toutlet   : {0}\n".format(self._idxcell_outlet)
        str += "\tinlets   : {0}\n".format(self._idxinlets)

        narea = 0
        try:
            narea = len(self.idxcells_area)
        except Exception:
            pass
        str += "\tarea     : {0} cells\n".format(narea)

        nboundary = 0
        try:
            nboundary = len(self.idxcells_boundary)
        except Exception:
            pass
        str += "\tboundary : {0} cells\n".format(nboundary)

        return str

    def __add__(self, other):
        """ Combine two catchment areas """
        catchment = self.clone()
        catchment.name = self.name + "+" + other.name
        catchment._idxcell_outlet = None
        catchment._idxinlets = None
        catchment._idxcells_boundary = None
        catchment._xycells_boundary = None
        if self._idxcells_area is not None \
           and other._idxcells_area is not None:
            catchment._idxcells_area = np.union1d(self.idxcells_area_filled,
                                                  other.idxcells_area_filled)
        return catchment

    def __sub__(self, other):
        """ Substract two catchment areas """
        catchment = self.clone()
        catchment.name = self.name + "-" + other.name
        catchment._idxcell_outlet = None
        catchment._idxinlets = None
        catchment._idxcells_boundary = None
        catchment._xycells_boundary = None

        # Difference between filled cell boundary
        catchment._idxcells_area = np.setdiff1d(self.idxcells_area_filled,
                                                other.idxcells_area_filled)

        return catchment

    @classmethod
    def from_dict(cls, dic):
        """ Create catchment from dictionary

        Parameters
        -----------
        dic : dict
            Dictionary containing catchment attributes

        Returns
        -----------
        catchment : hydrodiy.grid.Catchment
            Catchment instance
        """
        flowdir = Grid.from_dict(dic["flowdir"])
        catchment = Catchment(dic["name"], flowdir)
        catchment._idxcell_outlet = dic["idxcell_outlet"]
        catchment._idxintlets = dic["idxinlets"]

        area = np.array(dic["idxcells_area"]).astype(np.int64)
        catchment._idxcells_area = area

        filled = np.array(dic["idxcells_area_filled"]).astype(np.int64)
        catchment._idxcells_area_filled = filled

        return catchment

    @property
    def idxcell_outlet(self):
        """ Get outlet cell """
        if self._idxcell_outlet is None:
            errmess = "idxcell_outlet is None, please"\
                      + " set outlet."
            raise ValueError(errmess)

        return self._idxcell_outlet

    @property
    def idxinlets(self):
        """ Get inlet cells """
        return self._idxinlets

    @property
    def idxcells_area(self):
        """ Get cells of catchment area """
        if self._idxcells_area is None:
            errmess = "idxcells_area is None, please"\
                      + " delineate the area."
            raise ValueError(errmess)

        return self._idxcells_area

    @property
    def idxcells_area_filled(self):
        """ Get cells of filled catchment area (no holes)"""
        if self._idxcells_area_filled is None:
            errmess = "idxcells_area_filled is None, please"\
                      + " delineate the area."
            raise ValueError(errmess)
        return self._idxcells_area_filled

    @property
    def flowpathlengths(self):
        """ Get flowpaths cells expressed in number of cells """
        return self._flowpathlengths

    @property
    def xycells_boundary(self):
        """ Get xy coords of catchment boundary """
        return self._xycells_boundary

    @property
    def idxcells_boundary(self):
        """ Get cells of catchment boundary """
        if self._idxcells_boundary is None:
            errmess = "idxcells_boundary is None, please"\
                      + " delineate the area."
            raise ValueError(errmess)

        return self._idxcells_boundary

    @property
    def flowdir(self):
        """ Get flow direction grid """
        return self._flowdir

    def clone(self):
        """ Clone the current catchment """
        catchment = copy.deepcopy(self)
        return catchment

    def extent(self):
        """ Get catchment area extent """
        xy = self._flowdir.cell2coord(self._idxcells_area_filled)
        cz = self.flowdir.cellsize
        ext = (np.min(xy[:, 0])-cz/2, np.max(xy[:, 0])+cz/2,
               np.min(xy[:, 1])-cz/2, np.max(xy[:, 1])+cz/2)
        return ext

    def upstream(self, idxdown):
        """ Get upstream cell of a given cell """
        has_c_module("gis")

        idxdown = np.atleast_1d(idxdown).astype(np.int64)
        idxup = np.zeros((len(idxdown), 9), dtype=np.int64)
        ierr = c_hydrodiy_gis.upstream(FLOWDIRCODE,
                                       self._flowdir.data,
                                       idxdown,
                                       idxup)
        if ierr > 0:
            errmess = f"c_hydrodiy_gis.upstream returns {ierr}."
            raise ValueError(errmess)

        return idxup

    def downstream(self, idxup):
        """ Get downstream cell of a given cell """
        has_c_module("gis")

        idxup = np.atleast_1d(idxup).astype(np.int64)
        idxdown = np.zeros(len(idxup), dtype=np.int64)

        ierr = c_hydrodiy_gis.downstream(FLOWDIRCODE,
                                         self._flowdir.data,
                                         idxup,
                                         idxdown)
        if ierr > 0:
            errmess = f"c_hydrodiy_gis.downstream returns {ierr}."
            raise ValueError(errmess)

        return idxdown

    def delineate_area(self, idxcell_outlet, idxinlets=None, nval=1000000):
        """ Delineate catchment area from flow direction grid

        Parameters
        -----------
        idxcell_outlet : int
            Index of outlet cell
        idxinlets : list
            Index of inlet cells
        nval : int
            Maximum number of cells in area
        """
        has_c_module("gis")

        self._idxcell_outlet = np.int64(idxcell_outlet)

        if idxinlets is None:
            idxinlets = -1*np.ones(0, dtype=np.int64)
        else:
            idxinlets = np.atleast_1d(idxinlets).astype(np.int64)
            self._idxinlets = idxinlets

        idxcells = -1*np.ones(nval, dtype=np.int64)
        buffer1 = -1*np.ones(nval, dtype=np.int64)
        buffer2 = -1*np.ones(nval, dtype=np.int64)

        # Compute area
        flowdir = self.flowdir
        ierr = c_hydrodiy_gis.delineate_area(FLOWDIRCODE,
                                             flowdir.data,
                                             self.idxcell_outlet,
                                             idxinlets,
                                             idxcells,
                                             buffer1,
                                             buffer2)

        if ierr > 0:
            self._idxcells_area = None
            self._idxcells_area_filled = None
            errmess = f"c_hydrodiy_gis.delineate_area returns {ierr}."\
                      + f" Consider increasing buffer size ({nval})."
            raise ValueError(errmess)

        idx = idxcells >= 0
        self._idxcells_area = idxcells[idx]

        if idx.sum() > 0:
            # Fill area cells
            # .. create a minimal rectangle containing catchment area
            rowcol = flowdir.cell2rowcol(self._idxcells_area)
            i0, j0 = np.maximum(0, rowcol.min(axis=0)-1)
            i1, j1 = rowcol.max(axis=0)+1
            i1 = min(flowdir.nrows-1, i1)
            j1 = min(flowdir.ncols-1, j1)
            nrows, ncols = i1-i0+1, j1-j0+1
            grid = np.zeros((nrows, ncols), dtype=int)
            grid[rowcol[:, 0]-i0, rowcol[:, 1]-j0] = 1
            # .. fill holes
            grid = binary_fill_holes(grid)
            # .. get cell coordinates
            irows, icols = np.where(grid == 1)
            irows += i0
            icols += j0
            filled = irows*flowdir.ncols+icols
            self._idxcells_area_filled = filled
        else:
            self._idxcells_area_filled = self._idxcells_area

    def delineate_boundary(self, catchment_area_mask=None):
        """ Delineate catchment boundary from area """
        has_c_module("gis")

        if self._idxcells_area is None:
            errmess = "idxcells_arer is None, please"\
                      + " delineate the area"
            raise ValueError(errmess)

        nrows = self._flowdir.nrows
        ncols = self._flowdir.ncols

        # Initialise boundary cells with vector of same length
        # than area. Use FILLED area cells!
        cells_area = self._idxcells_area_filled
        nval = np.int64(len(cells_area))
        idxcells_boundary = -1*np.ones(nval, dtype=np.int64)
        buf = -1*np.ones(nval, dtype=np.int64)

        # Initialise catchment area mask
        if catchment_area_mask is None:
            catchment_area_mask = np.zeros(nrows*ncols, dtype=np.int64)
            catchment_area_mask[cells_area] = 1

        # Use filled area (algorithm failed when there are holes)
        # Compute boundary with varying dmax
        # This point could be improved!!
        ierr = c_hydrodiy_gis.delineate_boundary(nrows, ncols,
                                                 cells_area,
                                                 buf, catchment_area_mask,
                                                 idxcells_boundary)

        if ierr > 0:
            errmess = f"c_hydrodiy_gis.delineate_boundary returns {ierr}."
            raise ValueError(errmess)

        idx = idxcells_boundary >= 0
        idxcells_boundary = idxcells_boundary[idx]

        # Generate coordinates for the boundary
        xy = self._flowdir.cell2coord(idxcells_boundary)

        if len(idxcells_boundary) >= 3:
            # Exclude zero area angles
            # to allow processing by GDAL
            nrows = len(xy)
            idxok = np.zeros(nrows, dtype=np.int64)
            deteps = np.std(xy)*1e-6
            ierr = c_hydrodiy_gis.exclude_zero_area_boundary(deteps,
                                                             xy, idxok)

            if ierr > 0:
                errmess = "c_hydrodiy_gis.exclude_zero_area_boundary"\
                          + f" returns {ierr}."
                raise ValueError(errmess)
        else:
            idxok = np.ones(len(idxcells_boundary))

        # Store boundaries
        idx = idxok == 1
        self._idxcells_boundary = idxcells_boundary[idx]
        self._xycells_boundary = xy[idx]

    def compute_flowpathlengths(self):
        """ Compute length of all flow paths in catchment. """
        has_c_module("gis")

        idxcells_area = self.idxcells_area

        # Initialise flowpaths data
        nval = np.int64(len(idxcells_area))
        flowpaths = np.zeros((nval, 3), dtype=np.float64)

        # Compute flow paths
        fun = c_hydrodiy_gis.delineate_flowpathlengths_in_catchment
        ierr = fun(self.idxcell_outlet, FLOWDIRCODE, self.flowdir.data,
                   self.idxcells_area, flowpaths)
        if ierr > 0:
            errmess = "c_hydrodiy_gis.delineate_flowpathlengths_in_catchment"\
                      f" returns {ierr}."
            raise ValueError(errmess)

        cols = ["idxcell_start", "idxcell_end", "length[cell]"]
        self._flowpathlengths = pd.DataFrame(flowpaths, columns=cols)

    def intersect(self, grid, filled=False):
        """ Intersect catchment area with other grid and compute
        the weight of each cell from the new grid falling into the
        catchment.

        Parameters
        -----------
        grid : hydrodiy.grid.Grid
            Input grid (a.g. rainfall data grid).
        filled : bool
            Used filled catchment area.

        Returns
        -----------
        area_grid : hydrodiy.grid.Grid
            Grid matrix. This is the subset of the input grid encapsulating
            the catchment area. The grid values are cell weights.
            The grid contains information about start/end rows and columns
            in the input grid. This information is useful to extract data
            from the input grid quickly.
        idxcells : numpy.ndarray
            List of cells from the input grid falling into the catchment.
        weights :  numpy.ndarray
            List of cell weights.

        """
        has_c_module("gis")

        if grid.cellsize < self.flowdir.cellsize*2:
            warnmess = f"Cellsize of intersecting grid ({grid.cellsize})"\
                       + " is smaller than twice the cellsize of flow"\
                       + f" directon grid ({self.flowdir.cellsize})."
            warnings.warn(warnmess)

        xll, yll, csz, nrows, ncols = grid._getsize()
        _, _, csz_area, _, _ = self.flowdir._getsize()

        cells = self._idxcells_area_filled if filled else self._idxcells_area
        xy_area = self.flowdir.cell2coord(cells)
        npoints = np.zeros((1,), dtype=np.int64)
        idxcells = np.zeros(nrows*ncols, dtype=np.int64)
        weights = np.zeros(nrows*ncols, dtype=np.float64)

        # Compute intersection
        ierr = c_hydrodiy_gis.intersect(nrows, ncols,
                                        xll, yll, csz, csz_area,
                                        xy_area, npoints, idxcells, weights)

        if ierr > 0:
            errmess = f"c_hydrodiy_gis.intersect returns {ierr}."
            raise ValueError(errmess)

        idxcells = idxcells[:npoints[0]]
        weights = weights[:npoints[0]]

        # Get coordinates of lower left point
        coords = grid.cell2coord(idxcells)
        axll = np.min(coords[:, 0])-grid.cellsize/2
        ayll = np.min(coords[:, 1])-grid.cellsize/2

        # Weight grid
        rowcols = grid.cell2rowcol(idxcells)
        rows = np.unique(rowcols[:, 0])
        row_start, row_end = np.min(rows), np.max(rows)
        anrows = row_end-row_start + 1

        cols = np.unique(rowcols[:, 1])
        col_start, col_end = np.min(cols), np.max(cols)
        ancols = col_end-col_start + 1

        weights_array = np.zeros((anrows, ancols))
        weights_array[(rowcols[:, 0]-row_start)[:, None],
                      (rowcols[:, 1]-col_start)[:, None]] = weights[:, None]

        # Generate grid object
        comment = "Area grid for catchment"\
                  + f"[{self.name}] intersected with grid [{grid.name}]."
        area_grid = Grid("area_grid",
                         ncols=ancols, nrows=anrows,
                         cellsize=grid.cellsize,
                         xllcorner=axll, yllcorner=ayll,
                         dtype=np.float64, nodata=0,
                         comment=comment)
        area_grid.data = weights_array

        # Set additional attributes to area grid
        # to keep info
        area_grid.set_parent_attributes(grid, row_start, row_end,
                                        col_start, col_end)

        return area_grid, idxcells, weights

    def isin(self, idxcell, filled=False):
        """ Check if a cell is within the catchment area """

        if self._idxcells_area is None:
            errmess = "idxcells_boundary is None, please delineate the area"
            raise ValueError(errmess)

        cells = self._idxcells_area_filled if filled else self._idxcells_area
        return idxcell in cells

    def compute_area(self, to_proj, from_proj=None):
        """ Compute catchment area in km2 using a projection and
            applying the Shoelace algorithm.
            See https://en.wikipedia.org/wiki/Shoelace_formula

        Parameters
        -----------
        to_proj : pyproj.Proj
            Target projection obtained from the pyproj package.
            For example for the GDA94, we have
            to_proj = pyproj.Proj("+init=EPSG:3112")

        from_proj : pyproj.Proj
            Projection of grid coordinates obtained from the pyproj package.
            If None, it is assumed that the original coordinates are not
            projected.

            For example for the GDa94, we have
            from_proj = pyproj.Proj("+init=EPSG:4326")

        Returns
        -----------
        area : float
            Catchment area in km2
        """
        idxb = self.idxcells_boundary
        if idxb is None:
            errmess = "idxcells_boundary is None, please delineate the area"
            raise ValueError(errmess)

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
        area = 0.5e-6 * np.abs(np.dot(x, np.roll(y, 1))
                               - np.dot(y, np.roll(x, 1)))

        return area

    def plot_area(self, ax, *args, **kwargs):
        """ Plot catchment area """

        if self._idxcells_area is None:
            errmess = "idxcells_boundary is None, please delineate the area"
            raise ValueError(errmess)

        filled = kwargs.get("filled", False)
        cells = self._idxcells_area_filled if filled else self._idxcells_area

        xy = self.flowdir.cell2coord(cells)
        ax.plot(xy[:, 0], xy[:, 1], *args, **kwargs)

    def plot_boundary(self, ax, *args, **kwargs):
        """ Plot catchment boundary """

        if self._idxcells_boundary is None:
            errmess = "idxcells_boundary is None, please delineate the area"
            raise ValueError(errmess)

        xy = self.flowdir.cell2coord(self._idxcells_boundary)
        ax.plot(xy[:, 0], xy[:, 1], *args, **kwargs)

    def to_dict(self):
        """ Export data to a dictionnary """

        if self._idxcells_area is None:
            errmess = "idxcells_area is None, please delineate the area"
            raise ValueError(errmess)

        inlets = None
        if self._idxinlets is not None:
            inlets = list(self._idxinlets)

        dic = {
            "name": self.name,
            "idxcell_outlet": self._idxcell_outlet,
            "idxinlets": inlets,
            "idxcells_area": list(self._idxcells_area),
            "idxcells_area_filled": list(self._idxcells_area_filled),
            "flowdir": self.flowdir.to_dict(),
        }
        return dic


def delineate_river(flowdir, idxupstream, nval=1000000):
    """ Delineate river upstream point and flow direction grid

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
    """
    has_c_module("gis")

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

    if ierr > 0:
        errmess = f"c_hydrodiy_gis.delineate_river returns {ierr}."
        raise ValueError(errmess)

    nval = npoints[0]
    data = pd.DataFrame(data[:nval],
                        columns=["dist", "dx", "dy", "x", "y"])
    data["idxcell"] = idxcells[:nval]

    return data


def accumulate(flowdir, to_accumulate=None, nprint=100,
               max_accumulated_cells=-1):
    """ Compute flow accumulation from the flow direction grid

    Parameters
    -----------
    flowdir : hydrodiy.gis.grid.Catchment
        Flow direction grid
    to_accumulate : hydrodiy.gis.grid.Catchment
        Grid to accumulate. Should have the same dimensions than
        flowdir.
    nprint : int
        Frequency of log print
    max_accumulated_cells : int
        Maximimum number of accumulated cells

    Returns
    -----------
    accumulation : hydrodiy.gis.grid.Catchment
        Accumulated field
    """
    has_c_module("gis")

    nprint = np.int64(nprint)

    if max_accumulated_cells == -1:
        max_accumulated_cells = flowdir.nrows * flowdir.ncols
    max_accumulated_cells = np.int64(max_accumulated_cells)

    # Convert flowdir
    flowdir.dtype = np.int64

    # Set accumulation field
    if to_accumulate is None:
        to_accumulate = flowdir.clone()
        to_accumulate.fill(1)

    to_accumulate.dtype = np.float64

    # Initiase the accumulation grid with 0 accumulation
    accumulation = to_accumulate.clone()

    # Compute accumulation
    ierr = c_hydrodiy_gis.accumulate(nprint,
                                     max_accumulated_cells,
                                     to_accumulate.nodata,
                                     FLOWDIRCODE,
                                     flowdir.data,
                                     to_accumulate.data,
                                     accumulation.data)

    if ierr > 0:
        errmess = f"c_hydrodiy_gis.accumulate returns {ierr}."
        raise ValueError(errmess)

    return accumulation


def voronoi(catchment, xypoints):
    """ Compute the Voronoi weights for a set of points close to a catchment

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
    """
    has_c_module("gis")
    if catchment._idxcells_area is None:
        errmess = "Catchment idxcells_area is None, "\
                  + "please delineate the area"
        raise ValueError(errmess)

    xll, yll, csz, nrows, ncols = catchment._flowdir._getsize()

    idxcells_area = np.array(catchment._idxcells_area).astype(np.int64)
    xypoints = np.atleast_2d(xypoints).astype(np.float64)
    weights = np.zeros(xypoints.shape[0]).astype(np.float64)

    ierr = c_hydrodiy_gis.voronoi(nrows, ncols, xll, yll, csz,
                                  idxcells_area, xypoints, weights)

    if ierr > 0:
        errmess = f"c_hydrodiy_gis.voronoi returns {ierr}."
        raise ValueError(errmess)

    return weights


def get_grid(name):
    """ Get reference gridss defined in
    Bureau of meteorology products

    Parameters
    -----------
    name : str
        Name of grid :
        - AWRAL : Grid used in AWRAL products
            This grid is also used by QLD/SILO products.
        - AWAP : Grid used in AWAP products
        - WATERDYN : Grid used in CSIRO waterdyn products
        - AWRAL_DRAINAGE : Grid used in AWRAL products with drainage divisions
        - DLCD : Grid used by Geoscience Australia DLCD product (land cover)
        - .. all AWRAL subgrids stored in the
            hydrodiy/hygis/data/AWRAL_SUBGRIDS.zip file.

    extract : bool
        Force extraction of zip data

    Returns
    -----------
    gr : hydrodiy.gis.grid.Grid
        Mask grid containing 1 for cells within the grid
        and 0 for cells outside the grid

    """
    expected_base = ["AWRAL", "AWAP", "WATERDYN", "DLCD"]
    expected_subgrids = list(AWRAL_SUBGRIDS.gridid)

    if name in expected_subgrids:
        # Process AWRAL subgrids
        idx = AWRAL_SUBGRIDS.gridid == name
        info = AWRAL_SUBGRIDS.loc[idx, :].iloc[0]

        # Generate file names
        fh = "{}/{}.hdr".format(info.entity_type,
                                re.sub("\\..*$", "", info.grid_file))

        # Reads data
        gr = Grid.from_zip(FZIP_AWRAL_SUBGRIDS, fh)

        return gr

    elif name in expected_base:
        # Generate file names
        fbase = "{0}_GRID".format(name)
        fzip = os.path.join(F_HYGIS_DATA, "{0}.zip".format(fbase))
        fhdr = "{0}.hdr".format(fbase)

        # Reads data
        gr = Grid.from_zip(fzip, fhdr)

    else:
        errmess = f"Expected name in {expected_base} or AWRAL subgrids id,"\
                  + f"got {name}."
        raise ValueError(errmess)

    return gr


def slope(flowdir, altitude, nprint=100):
    """ Compute flow accumulation from the flow direction grid """
    has_c_module("gis")
    nprint = np.int64(nprint)

    # Convert flowdir
    flowdir.dtype = np.int64
    altitude.dtype = np.float64

    # Initiase the slope grid with 0 accumulation
    slopeval = altitude.clone()
    slopeval.fill(slopeval.nodata)

    # Compute slope
    cellsize = np.float64(altitude.cellsize)

    ierr = c_hydrodiy_gis.slope(nprint, cellsize, FLOWDIRCODE,
                                flowdir.data, altitude.data,
                                slopeval.data)

    if ierr > 0:
        errmess = f"c_hydrodiy_gis.slope returns {ierr}."
        raise ValueError(errmess)

    return slopeval


def gsmooth(grid, mask=None, coastwin=50, sigma=5.,
            minval=-np.inf, eps=1e-6):
    ''' Smooth gridded value to improve look of map '''
    # Check coastwin param
    if coastwin < 10*sigma:
        warnmess = "Expected coastwin > 10 x sigma, got "\
                   + f"coastwin={coastwin:0.1f} sigma={sigma:0.1f}."
        warnings.warn(warnmess)

    # Set up smooth grid
    smooth = grid.clone(dtype=np.float64)
    z0 = grid.data.copy()

    # Cut mask
    if mask is not None:
        # Check grid and mask have the same geometry
        if not grid.same_geometry(mask):
            raise ValueError('Mask does not have the same '
                             + 'geometry than input grid')

        # Check mask as integer dtype
        if not issubclass(mask.data.dtype.type, np.integer):
            errmess = "Expected integer dtype for mask,"\
                      + f" got {mask.dtype}."
            raise ValueError(errmess)

        # Set mask
        z0[mask.data == 0] = np.nan

    # Gapfill with nearest neighbour
    idx = np.isnan(z0) | (z0 < minval-eps)
    z0[idx] = -np.inf

    z1 = maximum_filter(z0, size=coastwin, mode='nearest')
    z1[~idx] = z0[~idx]

    # Smooth
    z2 = gaussian_filter(z1, sigma=sigma, mode='nearest')

    # Final mask cut
    if mask is not None:
        z2[mask.data == 0] = np.nan

    # Store
    smooth.data = z2

    return smooth
