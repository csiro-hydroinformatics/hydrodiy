#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


/* ************** Core subroutine ******************************
* Compute the baseflow component based on Eckhardt algorithm
* Eckhardt K. (2005) How to construct recursive digital filters for baseflow separation. Hydrological processes 19:507-515.
*
* C code was translated from R code provided by
* Jose Manuel Tunqui Neira and Vazken Andreassian, IRSTEA
*
* nval : length of input vectors (number of values)
* timestep_type : 0=hourly, 1=daily
* thresh : percentage from which the base flow should be considered as total flow
* tau : characteristic drainage timescale (hours) -> to calculate this parameter see [2,3,4]
* BFI_max : see Eckhardt (2005)
* inputs : flow time series
* outputs : baseflow series
*
************************************************/
int c_eckhardt(int nval, int timestep_type,
	double thresh, double tau, double BFI_max,
	double* inputs,
	double* outputs)
{
	int i;
	double q, qtmp, bf1, bf2, timestep_length;
    double C1, C2, C3, alpha;

    /* Check params */
    if(timestep_type != 0 && timestep_type != 1)
        return EDOM;

    if(thresh <0 || thresh > 1)
        return EDOM;

    if(BFI_max <0 || BFI_max > 1)
        return EDOM;

    /* Time step duration in hours */
    timestep_length = timestep_type == 0 ? 1 : 24;

    /* Filter constants */
    alpha = exp(-timestep_length/tau);
    C1 = (1 - BFI_max)*alpha;
    C2 = (1 - alpha)*BFI_max;
    C3 = 1 - (alpha*BFI_max);

    /* Initisalise */
    q = inputs[0];
    q = (q<0 || isnan(q)) ? 0 : inputs[0];
    bf1 = C2*q/C3;
    outputs[0] = bf1;

    /* loop through data */
    for(i=1; i<nval; i++)
    {
        /* Check inputs is valid */
        qtmp = inputs[i];
        q = (qtmp>=0 && !isnan(qtmp)) ? qtmp : q;

	    /* baseflow values */
        bf1 = (C1*bf1 + C2*q)/C3;
        bf2 = bf1 > q ? q : bf1;
        bf2 = bf1 > thresh*q ? q : bf1;

        /* Store data */
	    outputs[i] = bf2;
    }

    return 0;
}
