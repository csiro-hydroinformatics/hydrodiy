#ifndef __HY_DATA_BASEFLOW__
#define __HY_DATA_BASEFLOW__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

int c_eckhardt(int nval, int timestep_type,
	double thresh, double tau, double BFI_max,
	double* inputs,
	double* outputs);

#endif
