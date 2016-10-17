#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define GRID_ERROR 50000

double clipd(double x, double x0, double x1);
long clipi(long x, long x0, long x1);

long getnxy(long ncols, long idxcell, long *nxy);

long c_coord2cell(long nrows, long ncols,
    double xll, double yll, double csz,
    long nval, double * xycoords, long * idxcell);


long c_cell2coord(long nrows, long ncols,
    double xll, double yll, double csz,
    long nval, long * idxcell, double * xycoords);


long c_slice(long nrows, long ncols,
    double xll, double yll, double csz, double* data,
    long nval, double* xyslice, double * zslice);

long c_neighbours(long nrows, long ncols,
    long idxcell, long * neighbours);
