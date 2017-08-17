#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define GRID_ERROR 50000

double clipd(double x, double x0, double x1);
long long clipi(long long x, long long x0, long long x1);

long long getnxy(long long ncols, long long idxcell, long long *nxy);

long long c_coord2cell(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, double * xycoords, long long * idxcell);


long long c_cell2coord(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, long long * idxcell, double * xycoords);


long long c_slice(long long nrows, long long ncols,
    double xll, double yll, double csz, double* data,
    long long nval, double* xyslice, double * zslice);

long long c_neighbours(long long nrows, long long ncols,
    long long idxcell, long long * neighbours);
