#ifndef __HY_STAT_ARMODEL__
#define __HY_STAT_ARMODEL__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define ARMODEL_ERROR 56000

#define ARMODEL_NPARAMSMAX 10

double get_nan(void);

int c_armodel_sim(int nval, int ncols, int nparams,
        int fillnan,
        double simini, double * params,
        double * innov, double* outputs);

int c_armodel_residual(int nval, int ncols, int nparams,
        int fillnan,
        double stateini, double * params,
        double * inputs, double* residuals);

#endif
