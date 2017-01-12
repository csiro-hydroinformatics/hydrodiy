#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

int c_ar1innov(int nval, int ncol, double * params,
        double * innov, double* outputs);

int c_ar1inverse(int nval, int ncol, double * params,
        double * inputs, double* innov);

