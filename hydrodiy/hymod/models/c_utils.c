#include "c_utils.h"


int c_utils_getesize(void)
{
    return MODEL_ESIZE;
}

double c_utils_minmax(double min, double max, double input)
{
    return input < min ? min : 
            input > max ? max : input;
}

