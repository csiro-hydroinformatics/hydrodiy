#ifndef __HY_STAT_PARETOFRONT__
#define __HY_STAT_PARETOFRONT__


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

/**************
 * Identifies points that are dominated (i.e. not part of the pareto front)
 * nval : number of points
 * ncol : number of dimensions
 * orientation : orientation of the front (-1=negative, 1=positive)
 * data : points array [nvalxncol]
 * isdominated : list of dominated points (=1). Pareto front corresponds to =0.
**/
int c_paretofront(int nval,int ncol,
    int orientation,
    double* data,
    int* isdominated);

#endif
