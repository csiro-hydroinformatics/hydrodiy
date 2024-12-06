import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_points_inside_polygon.h':
    int c_inside(int nprint, int npoints, double * points,
        int nvertices, double * polygon,
        double atol,
        double * polygon_xlim, double * polygon_ylim,
        int * inside)


cdef extern from 'c_grid.h':
    long long c_coord2cell(long long nrows, long long ncols,
        double xll, double yll, double csz,
        long long nval, double * xycoords, long long * idxcell)

    long long c_cell2rowcol(long long nrows, long long ncols,
        long long nval, long long * idxcell, long long * rowcols)

    long long c_cell2coord(long long nrows, long long ncols,
        double xll, double yll, double csz,
        long long nval, long long * idxcell, double * xycoords)

    long long c_slice(long long nrows, long long ncols,
        double xll, double yll, double csz, double* data,
        long long nval, double* xyslice, double * zslice)

    long long c_neighbours(long long nrows, long long ncols,
        long long idxcell, long long * neighbours)

    long long c_upstream(long long nrows, long long ncols,
        long long * flowdircode, long long * flowdir,
        long long nval, long long * idxdown, long long * idxup)

    long long c_downstream(long long nrows, long long ncols,
        long long * flowdircode, long long * flowdir,
        long long nval, long long * idxup, long long * idxdown)

    long long c_accumulate(long long nrows, long long ncols,
        long long nprint, long long max_accumulated_cells,
        double nodata_to_accumulate,
        long long * flowdircode,
        long long * flowdir,
        double * to_accumulate,
        double * accumulation)

    long long c_intersect(long long nrows, long long ncols,
        double xll, double yll, double csz, double csz_area,
        long long nval, double * xy_area,
        long long ncells, long long * npoints,
        long long * idxcells, double * weights)

    long long c_voronoi(long long nrows, long long ncols,
        double xll, double yll, double csz,
        long long ncells, long long * idxcells_area,
        long long npoints, double * xypoints,
        double * weights)

    long long c_slope(long long nrows,
        long long ncols,
        long long nprint,
        double cellsize,
        long long * flowdircode,
        long long * flowdir,
        double * altitude,
        double * slopeval)



cdef extern from 'c_catchment.h':
    long long c_delineate_area(long long nrows, long long ncols,
        long long* flowdircode, long long * flowdir,
        long long idxoutlet,
        long long ninlets, long long * idxinlets,
        long long nval, long long * idxcells_area,
        long long * buffer1, long long * buffer2)

    long long c_delineate_boundary(long long nrows, long long ncols,
        long long nval,
        long long * idxcells_area,
        long long * buffer,
        long long * catchment_area_mask,
        long long * idxcells_boundary)

    long long c_exclude_zero_area_boundary(long long nval,
        double deteps, double * xycoords, long long * idxok)

    long long c_delineate_river(long long nrows, long long ncols,
        double xll, double yll, double csz,
        long long* flowdircode, long long * flowdir,
        long long idxupstream,
        long long nval, long long * npoints,
        long long * idxcells,
        double * data)

    long long c_delineate_flowpathlengths_in_catchment(long long nrows,
        long long ncols,
        long long * flowdircode,
        long long * flowdir,
        long long nval,
        long long * idxcells_area,
        long long idxcell_outlet,
        double * flowpathlengths)


def __cinit__(self):
    pass

def coord2cell(long long nrows, long long ncols, double xll, double yll,
        double csz,
		np.ndarray[double, ndim=2, mode='c'] xycoords not None,
        np.ndarray[long long, ndim=1, mode='c'] idxcell not None):

    cdef long long ierr

    # check dimensions
    assert xycoords.shape[0] == idxcell.shape[0]

    ierr = c_coord2cell(nrows, ncols, xll, yll, csz,
			xycoords.shape[0],
            <double*> np.PyArray_DATA(xycoords),
            <long long*> np.PyArray_DATA(idxcell))

    return ierr


def cell2coord(long long nrows, long long ncols,
        double xll, double yll, double csz,
        np.ndarray[long long, ndim=1, mode='c'] idxcell not None,
	np.ndarray[double, ndim=2, mode='c'] coords not None):

    cdef long long ierr

    # check dimensions
    assert coords.shape[0] == idxcell.shape[0]
    assert coords.shape[1] == 2

    ierr = c_cell2coord(nrows, ncols, xll, yll, csz,
                    coords.shape[0],
                    <long long*> np.PyArray_DATA(idxcell),
                    <double*> np.PyArray_DATA(coords))

    return ierr

def cell2rowcol(long long nrows, long long ncols,
        np.ndarray[long long, ndim=1, mode='c'] idxcell not None,
	np.ndarray[long long, ndim=2, mode='c'] rowcols not None):

    cdef long long ierr

    # check dimensions
    assert rowcols.shape[0] == idxcell.shape[0]
    assert rowcols.shape[1] == 2

    ierr = c_cell2rowcol(nrows, ncols,
                    rowcols.shape[0],
                    <long long*> np.PyArray_DATA(idxcell),
                    <long long*> np.PyArray_DATA(rowcols))

    return ierr


def cell2rowcol(long long nrows, long long ncols,
        np.ndarray[long long, ndim=1, mode='c'] idxcell not None,
		np.ndarray[long long, ndim=2, mode='c'] rowcols not None):

    cdef long long ierr

    # check dimensions
    assert rowcols.shape[0] == idxcell.shape[0]
    assert rowcols.shape[1] == 2

    ierr = c_cell2rowcol(nrows, ncols,
			rowcols.shape[0],
            <long long*> np.PyArray_DATA(idxcell),
            <long long*> np.PyArray_DATA(rowcols))

    return ierr


def slice(double xll, double yll, double csz,
        np.ndarray[double, ndim=2, mode='c'] data not None,
		np.ndarray[double, ndim=2, mode='c'] xyslice not None,
		np.ndarray[double, ndim=1, mode='c'] zslice not None):

    cdef long long ierr

    # check dimensions
    assert xyslice.shape[1] == 2

    ierr = c_slice(data.shape[0], data.shape[1], xll, yll, csz,
            <double*> np.PyArray_DATA(data),
			xyslice.shape[0],
            <double*> np.PyArray_DATA(xyslice),
            <double*> np.PyArray_DATA(zslice))

    return ierr


def neighbours(long long nrows, long long ncols, long long idxcell,
            np.ndarray[long long, ndim=1, mode='c'] neighbours not None):

    cdef long long ierr

    # check dimensions
    assert neighbours.shape[0] == 9

    ierr = c_neighbours(nrows, ncols, idxcell,
            <long long*> np.PyArray_DATA(neighbours))

    return ierr


def upstream(np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long long, ndim=1, mode='c'] idxdown not None,
            np.ndarray[long long, ndim=2, mode='c'] idxup not None):

    cdef long long ierr

    # check dimensions
    assert idxup.shape[0] == idxdown.shape[0]
    assert idxup.shape[1] == 9
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    ierr = c_upstream(flowdir.shape[0], flowdir.shape[1],
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            idxdown.shape[0],
            <long long*> np.PyArray_DATA(idxdown),
            <long long*> np.PyArray_DATA(idxup))

    return ierr


def downstream(np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long long, ndim=1, mode='c'] idxup not None,
            np.ndarray[long long, ndim=1, mode='c'] idxdown not None):

    cdef long long ierr

    # check dimensions
    assert idxup.shape[0] == idxdown.shape[0]
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    ierr = c_downstream(flowdir.shape[0], flowdir.shape[1],
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            idxup.shape[0],
            <long long*> np.PyArray_DATA(idxup),
            <long long*> np.PyArray_DATA(idxdown))

    return ierr


def delineate_area(np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            long long idxoutlet,
            np.ndarray[long long, ndim=1, mode='c'] idxinlets not None,
            np.ndarray[long long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[long long, ndim=1, mode='c'] buffer1 not None,
            np.ndarray[long long, ndim=1, mode='c'] buffer2 not None):

    cdef long long ierr

    # check dimensions
    assert idxcells_area.shape[0] == buffer1.shape[0]
    assert idxcells_area.shape[0] == buffer2.shape[0]
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3


    ierr = c_delineate_area(flowdir.shape[0], flowdir.shape[1],
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            idxoutlet,
            idxinlets.shape[0],
            <long long*> np.PyArray_DATA(idxinlets),
            idxcells_area.shape[0],
            <long long*> np.PyArray_DATA(idxcells_area),
            <long long*> np.PyArray_DATA(buffer1),
            <long long*> np.PyArray_DATA(buffer2))

    return ierr


def delineate_boundary(long long nrows, long long ncols,
            np.ndarray[long long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[long long, ndim=1, mode='c'] buffer not None,
            np.ndarray[long long, ndim=1, mode='c'] catchment_area_mask not None,
            np.ndarray[long long, ndim=1, mode='c'] idxcells_boundary not None):

    cdef long long ierr

    # check dimensions
    assert idxcells_area.shape[0] == idxcells_boundary.shape[0]
    assert idxcells_area.shape[0] == buffer.shape[0]
    assert nrows*ncols == catchment_area_mask.shape[0]

    ierr = c_delineate_boundary(nrows, ncols,
            idxcells_area.shape[0],
            <long long*> np.PyArray_DATA(idxcells_area),
            <long long*> np.PyArray_DATA(buffer),
            <long long*> np.PyArray_DATA(catchment_area_mask),
            <long long*> np.PyArray_DATA(idxcells_boundary))

    return ierr


def exclude_zero_area_boundary(double deteps,
            np.ndarray[double, ndim=2, mode='c'] xycoords not None,
            np.ndarray[long long, ndim=1, mode='c'] idxok not None):

    cdef long long ierr

    # check dimensions
    assert xycoords.shape[0] == idxok.shape[0]

    ierr = c_exclude_zero_area_boundary(xycoords.shape[0], deteps,
            <double*> np.PyArray_DATA(xycoords),
            <long long*> np.PyArray_DATA(idxok))

    return ierr



def delineate_river(double xll, double yll, double csz,
            np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            long long idxupstream,
            np.ndarray[long long, ndim=1, mode='c'] npoints not None,
            np.ndarray[long long, ndim=1, mode='c'] idxcells not None,
            np.ndarray[double, ndim=2, mode='c'] data not None):

    cdef long long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3
    assert data.shape[0] == idxcells.shape[0]
    assert data.shape[1] == 5
    assert npoints.shape[0] == 1

    ierr = c_delineate_river(flowdir.shape[0], flowdir.shape[1],
            xll, yll, csz,
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            idxupstream, idxcells.shape[0],
            <long long*> np.PyArray_DATA(npoints),
            <long long*> np.PyArray_DATA(idxcells),
            <double*> np.PyArray_DATA(data))

    return ierr


def accumulate(long long nprint, long long max_accumulated_cells,
            double nodata_to_accumulate,
            np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[double, ndim=2, mode='c'] to_accumulate not None,
            np.ndarray[double, ndim=2, mode='c'] accumulation not None):

    cdef long long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    assert to_accumulate.shape[0] == flowdir.shape[0]
    assert to_accumulate.shape[1] == flowdir.shape[1]

    assert accumulation.shape[0] == flowdir.shape[0]
    assert accumulation.shape[1] == flowdir.shape[1]


    ierr = c_accumulate(flowdir.shape[0], flowdir.shape[1],
            nprint, max_accumulated_cells,
            nodata_to_accumulate,
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            <double*> np.PyArray_DATA(to_accumulate),
            <double*> np.PyArray_DATA(accumulation))

    return ierr


def intersect(long long nrows, long long ncols,
            double xll, double yll, double csz, double csz_area,
            np.ndarray[double, ndim=2, mode='c'] xy_area not None,
            np.ndarray[long long, ndim=1, mode='c'] npoints not None,
            np.ndarray[long long, ndim=1, mode='c'] idxcells not None,
            np.ndarray[double, ndim=1, mode='c'] weights not None):

    cdef long long ierr

    # check dimensions
    assert xy_area.shape[1] == 2
    assert npoints.shape[0] == 1

    assert idxcells.shape[0] == weights.shape[0]

    ierr = c_intersect(nrows, ncols,
            xll, yll, csz, csz_area, xy_area.shape[0],
            <double*> np.PyArray_DATA(xy_area),
            idxcells.shape[0],
            <long long*> np.PyArray_DATA(npoints),
            <long long*> np.PyArray_DATA(idxcells),
            <double*> np.PyArray_DATA(weights))

    return ierr


def voronoi(long long nrows, long long ncols,
            double xll, double yll, double csz,
            np.ndarray[long long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[double, ndim=2, mode='c'] xypoints not None,
            np.ndarray[double, ndim=1, mode='c'] weights not None):

    cdef long long ierr

    # check dimensions
    assert xypoints.shape[1] == 2
    assert xypoints.shape[0] == weights.shape[0]

    ierr = c_voronoi(nrows, ncols,
            xll, yll, csz, idxcells_area.shape[0],
            <long long*> np.PyArray_DATA(idxcells_area),
            xypoints.shape[0],
            <double*> np.PyArray_DATA(xypoints),
            <double*> np.PyArray_DATA(weights))

    return ierr


def slope(long long nprint, double cellsize,
            np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[double, ndim=2, mode='c'] altitude not None,
            np.ndarray[double, ndim=2, mode='c'] slopeval not None):

    cdef long long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    assert altitude.shape[0] == flowdir.shape[0]
    assert altitude.shape[1] == flowdir.shape[1]

    assert slopeval.shape[0] == flowdir.shape[0]
    assert slopeval.shape[1] == flowdir.shape[1]

    # Run C code
    ierr = c_slope(flowdir.shape[0], flowdir.shape[1],
            nprint, cellsize,
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            <double*> np.PyArray_DATA(altitude),
            <double*> np.PyArray_DATA(slopeval))

    return ierr


def points_inside_polygon(double atol, int nprint,
            np.ndarray[double, ndim=2, mode='c'] points not None,
            np.ndarray[double, ndim=2, mode='c'] polygon not None,
            np.ndarray[int, ndim=1, mode='c'] inside not None):

    cdef int ierr
    cdef double polygon_xlim[2]
    cdef double polygon_ylim[2]

    # check dimensions
    assert points.shape[0] == inside.shape[0]
    assert points.shape[1] == 2
    assert polygon.shape[1] == 2

    # Define polygon extension
    polygon_xlim[0] = polygon[:, 0].min()
    polygon_xlim[1] = polygon[:, 0].max()

    polygon_ylim[0] = polygon[:, 1].min()
    polygon_ylim[1] = polygon[:, 1].max()

    # Run C code
    ierr = c_inside(nprint, points.shape[0],
            <double*> np.PyArray_DATA(points),
            polygon.shape[0],
            <double*> np.PyArray_DATA(polygon),
            atol, polygon_xlim, polygon_ylim,
            <int*> np.PyArray_DATA(inside))

    return ierr



def delineate_flowpathlengths_in_catchment(long long idxcell_outlet,
            np.ndarray[long long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[double, ndim=2, mode='c'] flowpathlengths not None):

    cdef long long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3
    assert flowpathlengths.shape[0] == idxcells_area.shape[0]
    assert flowpathlengths.shape[1] == 3

    ierr = c_delineate_flowpathlengths_in_catchment(
            flowdir.shape[0], flowdir.shape[1],
            <long long*> np.PyArray_DATA(flowdircode),
            <long long*> np.PyArray_DATA(flowdir),
            idxcells_area.shape[0],
            <long long*> np.PyArray_DATA(idxcells_area),
            idxcell_outlet,
            <double*> np.PyArray_DATA(flowpathlengths))

    return ierr

