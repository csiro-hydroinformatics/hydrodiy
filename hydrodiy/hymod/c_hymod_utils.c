#include "c_hymod_utils.h"


int c_hymod_getesize(void)
{
    return HYMOD_ESIZE;
}

double c_hymod_minmax(double min, double max, double input)
{
    return input < min ? min : 
            input > max ? max : input;
}

