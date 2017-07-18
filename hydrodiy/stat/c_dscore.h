#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define DSCORE_EPS 1e-8

int c_ensrank(int nval, int ncol, double* sim,
        double * fmat, double * ranks);

