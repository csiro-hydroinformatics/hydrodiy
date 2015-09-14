
#ifndef __HYMOD_UTILS__
#define __HYMOD_UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message for vector size errors */
#define HYMOD_ESIZE 500

int c_hymod_getesize(void);

double c_hymod_minmax(double min,double max,double input);

#endif
