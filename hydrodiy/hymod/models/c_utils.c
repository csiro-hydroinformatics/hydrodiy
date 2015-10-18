#include "c_utils.h"


int c_utils_getesize(int * esize)
{
    esize[0] = ESIZE_INPUTS;
    esize[1] = ESIZE_OUTPUTS;
    esize[2] = ESIZE_PARAMS; 
    esize[3] = ESIZE_STATES;
    esize[4] = ESIZE_STATESUH;

    return 0;
}

double c_utils_minmax(double min, double max, double input)
{
    return input < min ? min : 
            input > max ? max : input;
}

