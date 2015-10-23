
#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message for vector size errors */
#define ESIZE_OUTPUTS 500
#define ESIZE_INPUTS 600
#define ESIZE_PARAMS 700
#define ESIZE_STATES 800
#define ESIZE_STATESUH 900

int c_utils_getesize(int * esize);

double c_utils_minmax(double min,double max,double input);

double c_utils_tanh(double x);

double c_utils_doublesqrt1(double x);

#endif
