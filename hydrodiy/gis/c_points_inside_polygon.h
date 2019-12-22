#ifndef __HY_GIS_PTSINSIDEPOLY__
#define __HY_GIS_PTSINSIDEPOLY__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define INSIDE_ERROR 60000

int c_inside(int nprint, int npoints, double * points,
    int nvertices, double * polygon,
    double atol,
    double * polygon_xlim, double * polygon_ylim,
    int * inside);

#endif
