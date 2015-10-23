#include "c_utils.h"


int c_utils_getesize(int * esize)
{
    esize[0] = ESIZE_INPUTS;
    esize[1] = ESIZE_OUTPUTS;
    esize[2] = ESIZE_PARAMS; 
    esize[3] = ESIZE_STATES;
    esize[4] = ESIZE_STATESUH;
    esize[5] = EMODEL_RUN;

    return 0;
}

double c_utils_minmax(double min, double max, double input)
{
    return input < min ? min : 
            input > max ? max : input;
}

double c_utils_tanh(double x)
{
    double a, b, xsq;
    x = x > 4.9 ? 4.9 : x;
    xsq = x*x;
    a = (((36.*xsq+6930.)*xsq+270270.)*xsq+2027025.)*x;
    b = (((xsq+630.)*xsq+51975.)*xsq+945945.)*xsq+2027025.;
    return a/b;
}

