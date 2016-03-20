#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

int c_crps(int nval,int ncol,
    int use_weights, int is_sorted,
    double* obs,
    double* sim, 
    double* weights_vector,
    double* reliability_table,
    double* crps_decompos);

