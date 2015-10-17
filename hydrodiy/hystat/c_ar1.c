#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = ar1 x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 paramter
*   params[1] : initial value of ar1 process
* innov = innovation series e[i]
* output = model output
*
************************************************/
int c_ar1innov(int nval, double * params, 
        double * innov, double* output)
{
	int i;
    double ar1, y0;

    /* Get parameters */
    ar1 = params[0];
    y0 = params[1];

    /* loop through data */
    for(i=0; i<nval; i++){
        y0 = ar1*y0 +innov[i];
        output[i] = y0;
    }
    
    return 0;
}

/* ************** Core subroutine ******************************
* Computes innovation from ar 1 series :
*   y[i+1] = ar1 x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 parameter
*   params[1] : initial value of ar1 process
* input = ar1 timeseries
* innov = innovation series e[i]
*
************************************************/
int c_ar1inverse(int nval, double * params, 
        double * input, double* innov)
{
	int i;
    double ar1, y0;

    /* Get parameters */
    ar1 = params[0];
    y0 = params[1];

    /* loop through data */
    for(i=0; i<nval; i++){
        innov[i] = input[i]-ar1*y0;
        y0 = input[i];
    }
    
    return 0;
}
