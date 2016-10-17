#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "c_grid.h"

#define CATCHMENT_ERROR 60000

#define PI 3.14159265358979323846


long c_upstream(long nrows, long ncols,
    long * flowdircode,
    long * flowdir,
    long nval, long * idxdown, long * idxup);


long c_downstream(long nrows, long ncols,
    long * flowdircode,
    long * flowdir,
    long nval, long * idxup, long * idxdown);


long c_delineate_area(long nrows, long ncols,
    long* flowdircode,
    long * flowdir,
    long idxoutlet,
    long ninlets, long * idxinlets,
    long nval, long * idxcells,
    long * buffer1, long * buffer2);


long c_delineate_boundary(long nrows, long ncols,
    long nval,
    long * idxcells_area,
    long * buffer,
    long * grid_area,
    long * idxcells_boundary);

long c_delineate_river(long nrows, long ncols,
    double xll, double yll, double csz,
    long* flowdircode,
    long * flowdir,
    long idxupstream,
    long nval, long * npoints,
    long * idxcells,
    double * data);


long c_accumulate(long nrows, long ncols,
    long nprlong, long maxarea,
    long * flowdircode,
    long * flowdir,
    long * accumulation);


long c_intersect(long nrows, long ncols,
    double xll, double yll, double csz, double csz_area,
    long nval, double * xy_area,
    long ncells, long * npoints,
    long * idxcells, double * weights);


long c_voronoi(long nrows, long ncols,
    double xll, double yll, double csz,
    long ncells, long * idxcells_area,
    long npoints, double * xypoints,
    double * weights);


