import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_grid.h':
    long c_coord2cell(long nrows, long ncols,
        double xll, double yll, double csz,
        long nval, double * xycoords, long * idxcell)

    long c_cell2coord(long nrows, long ncols,
        double xll, double yll, double csz,
        long nval, long * idxcell, double * xycoords)

    long c_slice(long nrows, long ncols,
        double xll, double yll, double csz, double* data,
        long nval, double* xyslice, double * zslice)

    long c_neighbours(long nrows, long ncols,
        long idxcell, long * neighbours)


cdef extern from 'c_catchment.h':
    long c_upstream(long nrows, long ncols,
        long * flowdircode, long * flowdir,
        long nval, long * idxdown, long * idxup)

    long c_downstream(long nrows, long ncols,
        long * flowdircode, long * flowdir,
        long nval, long * idxup, long * idxdown)

    long c_delineate_area(long nrows, long ncols,
        long* flowdircode, long * flowdir,
        long idxoutlet,
        long ninlets, long * idxinlets,
        long nval, long * idxcells,
        long * buffer1, long * buffer2)

    long c_delineate_boundary(long nrows, long ncols,
        long nval,
        long * idxcells_area,
        long * buffer,
        long * grid_area,
        long * idxcells_boundary)

    long c_delineate_river(long nrows, long ncols,
        double xll, double yll, double csz,
        long* flowdircode, long * flowdir,
        long idxupstream,
        long nval, long * npoints,
        long * idxcells,
        double * data)

    long c_accumulate(long nrows, long ncols,
        long nprint, long maxarea,
        long * flowdircode,
        long * flowdir,
        long * accumulation)

    long c_intersect(long nrows, long ncols,
        double xll, double yll, double csz, double csz_area,
        long nval, double * xy_area,
        long ncells, long * npoints,
        long * idxcells, double * weights)

    long c_voronoi(long nrows, long ncols,
        double xll, double yll, double csz,
        long ncells, long * idxcells_area,
        long npoints, double * xypoints,
        double * weights)



def coord2cell(long nrows, long ncols, double xll, double yll,
        double csz,
		np.ndarray[double, ndim=2, mode='c'] xycoords not None,
        np.ndarray[long, ndim=1, mode='c'] idxcell not None):

    cdef long ierr

    # check dimensions
    assert xycoords.shape[0] == idxcell.shape[0]

    ierr = c_coord2cell(nrows, ncols, xll, yll, csz,
			xycoords.shape[0],
            <double*> np.PyArray_DATA(xycoords),
            <long*> np.PyArray_DATA(idxcell))

    return ierr


def cell2coord(long nrows, long ncols, double xll, double yll,
        double csz,
        np.ndarray[long, ndim=1, mode='c'] idxcell not None,
		np.ndarray[double, ndim=2, mode='c'] xycoords not None):

    cdef long ierr

    # check dimensions
    assert xycoords.shape[0] == idxcell.shape[0]
    assert xycoords.shape[1] == 2

    ierr = c_cell2coord(nrows, ncols, xll, yll, csz,
			xycoords.shape[0],
            <long*> np.PyArray_DATA(idxcell),
            <double*> np.PyArray_DATA(xycoords))

    return ierr


def slice(double xll, double yll, double csz,
        np.ndarray[double, ndim=2, mode='c'] data not None,
		np.ndarray[double, ndim=2, mode='c'] xyslice not None,
		np.ndarray[double, ndim=1, mode='c'] zslice not None):

    cdef long ierr

    # check dimensions
    assert xyslice.shape[1] == 2

    ierr = c_slice(data.shape[0], data.shape[1], xll, yll, csz,
            <double*> np.PyArray_DATA(data),
			xyslice.shape[0],
            <double*> np.PyArray_DATA(xyslice),
            <double*> np.PyArray_DATA(zslice))

    return ierr


def neighbours(long nrows, long ncols, long idxcell,
            np.ndarray[long, ndim=1, mode='c'] neighbours not None):

    cdef long ierr

    # check dimensions
    assert neighbours.shape[0] == 9

    ierr = c_neighbours(nrows, ncols, idxcell,
            <long*> np.PyArray_DATA(neighbours))

    return ierr


def upstream(np.ndarray[long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long, ndim=1, mode='c'] idxdown not None,
            np.ndarray[long, ndim=2, mode='c'] idxup not None):

    cdef long ierr

    # check dimensions
    assert idxup.shape[0] == idxdown.shape[0]
    assert idxup.shape[1] == 9
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    ierr = c_upstream(flowdir.shape[0], flowdir.shape[1],
            <long*> np.PyArray_DATA(flowdircode),
            <long*> np.PyArray_DATA(flowdir),
            idxdown.shape[0],
            <long*> np.PyArray_DATA(idxdown),
            <long*> np.PyArray_DATA(idxup))

    return ierr


def downstream(np.ndarray[long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long, ndim=1, mode='c'] idxup not None,
            np.ndarray[long, ndim=1, mode='c'] idxdown not None):

    cdef long ierr

    # check dimensions
    assert idxup.shape[0] == idxdown.shape[0]
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    ierr = c_downstream(flowdir.shape[0], flowdir.shape[1],
            <long*> np.PyArray_DATA(flowdircode),
            <long*> np.PyArray_DATA(flowdir),
            idxup.shape[0],
            <long*> np.PyArray_DATA(idxup),
            <long*> np.PyArray_DATA(idxdown))

    return ierr


def delineate_area(np.ndarray[long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long, ndim=2, mode='c'] flowdir not None,
            long idxoutlet,
            np.ndarray[long, ndim=1, mode='c'] idxinlets not None,
            np.ndarray[long, ndim=1, mode='c'] idxcells not None,
            np.ndarray[long, ndim=1, mode='c'] buffer1 not None,
            np.ndarray[long, ndim=1, mode='c'] buffer2 not None):

    cdef long ierr

    # check dimensions
    assert idxcells.shape[0] == buffer1.shape[0]
    assert idxcells.shape[0] == buffer2.shape[0]
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3


    ierr = c_delineate_area(flowdir.shape[0], flowdir.shape[1],
            <long*> np.PyArray_DATA(flowdircode),
            <long*> np.PyArray_DATA(flowdir),
            idxoutlet,
            idxinlets.shape[0],
            <long*> np.PyArray_DATA(idxinlets),
            idxcells.shape[0],
            <long*> np.PyArray_DATA(idxcells),
            <long*> np.PyArray_DATA(buffer1),
            <long*> np.PyArray_DATA(buffer2))

    return ierr


def delineate_boundary(long nrows, long ncols,
            np.ndarray[long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[long, ndim=1, mode='c'] buffer not None,
            np.ndarray[long, ndim=1, mode='c'] grid_area not None,
            np.ndarray[long, ndim=1, mode='c'] idxcells_boundary not None):

    cdef long ierr

    # check dimensions
    assert idxcells_area.shape[0] == idxcells_boundary.shape[0]
    assert idxcells_area.shape[0] == buffer.shape[0]
    assert nrows*ncols == grid_area.shape[0]

    ierr = c_delineate_boundary(nrows, ncols,
            idxcells_area.shape[0],
            <long*> np.PyArray_DATA(idxcells_area),
            <long*> np.PyArray_DATA(buffer),
            <long*> np.PyArray_DATA(grid_area),
            <long*> np.PyArray_DATA(idxcells_boundary))

    return ierr


def delineate_river(double xll, double yll, double csz,
            np.ndarray[long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long, ndim=2, mode='c'] flowdir not None,
            long idxupstream,
            np.ndarray[long, ndim=1, mode='c'] npoints not None,
            np.ndarray[long, ndim=1, mode='c'] idxcells not None,
            np.ndarray[double, ndim=2, mode='c'] data not None):

    cdef long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3
    assert data.shape[0] == idxcells.shape[0]
    assert data.shape[1] == 5
    assert npoints.shape[0] == 1

    ierr = c_delineate_river(flowdir.shape[0], flowdir.shape[1],
            xll, yll, csz,
            <long*> np.PyArray_DATA(flowdircode),
            <long*> np.PyArray_DATA(flowdir),
            idxupstream, idxcells.shape[0],
            <long*> np.PyArray_DATA(npoints),
            <long*> np.PyArray_DATA(idxcells),
            <double*> np.PyArray_DATA(data))

    return ierr


def accumulate(long nprint, long maxarea,
            np.ndarray[long, ndim=2, mode='c'] flowdircode not None,
            np.ndarray[long, ndim=2, mode='c'] flowdir not None,
            np.ndarray[long, ndim=2, mode='c'] accumulation not None):

    cdef long ierr

    # check dimensions
    assert flowdircode.shape[0] == 3
    assert flowdircode.shape[1] == 3

    assert accumulation.shape[0] == flowdir.shape[0]
    assert accumulation.shape[1] == flowdir.shape[1]


    ierr = c_accumulate(flowdir.shape[0], flowdir.shape[1],
            nprint, maxarea,
            <long*> np.PyArray_DATA(flowdircode),
            <long*> np.PyArray_DATA(flowdir),
            <long*> np.PyArray_DATA(accumulation))

    return ierr


def intersect(long nrows, long ncols,
            double xll, double yll, double csz, double csz_area,
            np.ndarray[double, ndim=2, mode='c'] xy_area not None,
            np.ndarray[long, ndim=1, mode='c'] npoints not None,
            np.ndarray[long, ndim=1, mode='c'] idxcells not None,
            np.ndarray[double, ndim=1, mode='c'] weights not None):

    cdef long ierr

    # check dimensions
    assert xy_area.shape[1] == 2
    assert npoints.shape[0] == 1

    assert idxcells.shape[0] == weights.shape[0]

    ierr = c_intersect(nrows, ncols,
            xll, yll, csz, csz_area, xy_area.shape[0],
            <double*> np.PyArray_DATA(xy_area),
            idxcells.shape[0],
            <long*> np.PyArray_DATA(npoints),
            <long*> np.PyArray_DATA(idxcells),
            <double*> np.PyArray_DATA(weights))

    return ierr


def voronoi(long nrows, long ncols,
            double xll, double yll, double csz,
            np.ndarray[long, ndim=1, mode='c'] idxcells_area not None,
            np.ndarray[double, ndim=2, mode='c'] xypoints not None,
            np.ndarray[double, ndim=1, mode='c'] weights not None):

    cdef long ierr

    # check dimensions
    assert xypoints.shape[1] == 2
    assert xypoints.shape[0] == weights.shape[0]

    ierr = c_voronoi(nrows, ncols,
            xll, yll, csz, idxcells_area.shape[0],
            <long*> np.PyArray_DATA(idxcells_area),
            xypoints.shape[0],
            <double*> np.PyArray_DATA(xypoints),
            <double*> np.PyArray_DATA(weights))

    return ierr

