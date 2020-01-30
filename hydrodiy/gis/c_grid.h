#ifndef __HY_GIS_GRID__
#define __HY_GIS_GRID__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define GRID_ERROR 50000

double clipd(double x, double x0, double x1);


long long clipi(long long x, long long x0, long long x1);


long long getnxy(long long ncols, long long idxcell, long long *nxy);

long long getcoord(long long nrows, long long ncols, double xll, double yll,
                double csz, long long idxcell, double *coord);

long long c_coord2cell(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, double * xycoords, long long * idxcell);


long long c_cell2rowcol(long long nrows, long long ncols,
    long long nval, long long * idxcell, long long * rowcols);


long long c_cell2coord(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, long long * idxcell, double * xycoords);


long long c_slice(long long nrows, long long ncols,
    double xll, double yll, double csz, double* data,
    long long nval, double* xyslice, double * zslice);


long long c_neighbours(long long nrows, long long ncols,
    long long idxcell, long long * neighbours);


long long c_upstream(long long nrows, long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval, long long * idxdown, long long * idxup);


long long c_downstream(long long nrows, long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval, long long * idxup, long long * idxdown);


long long c_accumulate(long long nrows, long long ncols,
    long long nprint, long long max_accumulated_cells,
    double nodata_to_accumulate,
    long long * flowdircode,
    long long * flowdir,
    double * to_accumulate,
    double * accumulation);


long long c_intersect(long long nrows, long long ncols,
    double xll, double yll, double csz, double csz_area,
    long long nval, double * xy_area,
    long long ncells, long long * npoints,
    long long * idxcells, double * weights);


long long c_voronoi(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long ncells, long long * idxcells_area,
    long long npoints, double * xypoints,
    double * weights);


long long c_slope(long long nrows,
    long long ncols,
    long long nprint,
    double cellsize,
    long long * flowdircode,
    long long * flowdir,
    double * altitude,
    double * slopeval);

#endif
