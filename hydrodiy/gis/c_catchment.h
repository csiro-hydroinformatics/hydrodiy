#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "c_grid.h"

#define CATCHMENT_ERROR 60000

#define PI 3.14159265358979323846

long long c_delineate_area(long long nrows, long long ncols,
    long long* flowdircode,
    long long * flowdir,
    long long idxoutlet,
    long long ninlets, long long * idxinlets,
    long long nval, long long * idxcells_area,
    long long * buffer1, long long * buffer2);


long long c_delineate_boundary(long long nrows, long long ncols,
    long long nval,
    long long * idxcells_area,
    long long * buffer,
    long long * catchment_area_mask,
    long long * idxcells_boundary);

long long c_exclude_zero_area_boundary(long long nval,
    double deteps, double * xycoords, long long * idxok);

long long c_delineate_river(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long* flowdircode,
    long long * flowdir,
    long long idxupstream,
    long long nval, long long * npoints,
    long long * idxcells,
    double * data);

long long c_delineate_flowpathlengths_in_catchment(long long nrows,
    long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval,
    long long * idxcells_area,
    long long idxcell_outlet,
    double * flowpathlengths);

