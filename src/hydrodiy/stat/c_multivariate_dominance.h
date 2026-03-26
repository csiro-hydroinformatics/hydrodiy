#ifndef __HY_STAT_MVTDOMINANCE__
#define __HY_STAT_MVTDOMINANCE__


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

/**************
 * Counts the number of dominated points
 * nval : number of points
 * ncol : number of dimensions
 * orientation : orientation of the front (-1=negative, 1=positive)
 * data : points array [nvalxncol]
 * ndominating : Number of dominated points.
**/
int c_multivariate_dominance(int nval,int ncol,
                             int orientation,
                             int printlog,
                             double* data,
                             int* ndominating);

#endif
