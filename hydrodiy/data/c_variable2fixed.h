#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "c_utils.h"

int c_variable2fixed(int nval,int ncol,
    int use_weights, int is_sorted,
    double* obs,
    double* sim,
    double* weights_vector,
    double* reliability_table,
    double* crps_decompos);

