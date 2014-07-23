#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int c_ar1random(int nval, double *params, 
        unsigned long int seed,  double* output);

int c_ar1innov(int nval, double * params, 
        double * innov, double* output);

int c_ar1inverse(int nval, double * params, 
        double * input, double* innov);

