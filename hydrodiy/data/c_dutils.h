#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Error code */
#define DUTILS_ERROR 110000


int c_aggregate(int nval, int oper, int maxnan, int * aggindex,
    double * inputs, double * outputs, int * iend);

long long c_combi(int n, int k);
