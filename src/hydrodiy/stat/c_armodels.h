#ifndef __HY_STAT_ARMODEL__
#define __HY_STAT_ARMODEL__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define ARMODEL_ERROR 56000

#define ARMODEL_NPARAMSMAX 10

int c_armodel_sim(int nval, int nparams,
        double sim_mean,
        double sim_ini, double * params,
        double * innov, double* outputs);

int c_armodel_residual(int nval, int nparams,
        double sim_mean,
        double sim_ini, double * params,
        double * inputs, double* residuals);

#endif
