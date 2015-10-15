
#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message for vector size errors */
#define MODEL_ESIZE 500

int c_utils_getesize(void);

double c_utils_minmax(double min,double max,double input);

#endif
