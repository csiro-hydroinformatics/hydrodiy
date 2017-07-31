#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define ESIZE 5000
#define EVALUE 5001

int c_ensrank(double eps, int nval, int ncol, double* sim,
        double * fmat, double * ranks);

