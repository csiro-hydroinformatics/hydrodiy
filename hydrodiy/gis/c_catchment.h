#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "c_grid.h"

#define CATCHMENT_ERROR 60000

#define PI 3.14159265358979323846


long long c_upstream(long long nrows, long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval, long long * idxdown, long long * idxup);


long long c_downstream(long long nrows, long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval, long long * idxup, long long * idxdown);


long long c_delineate_area(long long nrows, long long ncols,
    long long* flowdircode,
    long long * flowdir,
    long long idxoutlet,
    long long ninlets, long long * idxinlets,
    long long nval, long long * idxcells,
    long long * buffer1, long long * buffer2);


long long c_delineate_boundary(long long nrows, long long ncols,
    long long nval,
    long long * idxcells_area,
    long long * buffer,
    long long * grid_area,
    long long * idxcells_boundary);

long long c_delineate_river(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long* flowdircode,
    long long * flowdir,
    long long idxupstream,
    long long nval, long long * npoints,
    long long * idxcells,
    double * data);


long long c_accumulate(long long nrows, long long ncols,
    long long nprint, long long maxarea,
    long long * flowdircode,
    long long * flowdir,
    long long * accumulation);


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


