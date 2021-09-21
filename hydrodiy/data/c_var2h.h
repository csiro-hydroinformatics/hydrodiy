#ifndef __HY_DATA_VARH__
#define __HY_DATA_VARH__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Error code */
#define VAR2H_ERROR 130000

int c_var2h(int nvalvar, int nvalh,
    int starthour, int maxgapsec,
    long long * varsec,
    double * varvalues,
    long long hstartsec,
    double * hvalues);

#endif
