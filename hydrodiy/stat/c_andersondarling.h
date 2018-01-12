#include <stdio.h>
#include <math.h>

/* Routines from Marsaglia and Marsaglia (2014) */
#include "ADinf.h"
#include "AnDarl.h"


int c_ad_probexactinf(int nval, double *data, double *prob);

int c_ad_probn(int nval, int nsample, double *data, double *prob);

int c_ad_probapproxinf(int nval, double *data, double *prob);

