#ifndef __HY_DATA_QUALITCONTROL__
#define __HY_DATA_QUALITCONTROL__


#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Error code */
#define QUALITYCONTROL_ERROR 120000

int c_islin(int nval, double thresh, double tol, int npoints,
    double * data, int * islin);

#endif
