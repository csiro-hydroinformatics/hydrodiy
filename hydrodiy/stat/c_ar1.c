#include "c_ar1.h"

/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = ar1 x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 paramter
*   params[1] : initial value of ar1 process
* innov = innovation series e[i]
* outputs = model output
*
************************************************/
int c_ar1innov(int nval, int ncol, double * params,
        double * innov, double* outputs)
{
	int i, j;
    double alpha, y0;


    /* ar1 coefficient */
    alpha = params[0];

    /* loop across columns */
    for(j=0; j<ncol; j++)
    {
        /* Initialise AR1 */
        y0 = params[1];

        /* loop through values */
        for(i=0; i<nval; i++)
        {
            y0 = alpha*y0 + innov[ncol*i+j];
            outputs[ncol*i+j] = y0;
        }
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
* inputs = ar1 timeseries
* innov = innovation series e[i]
*
************************************************/
int c_ar1inverse(int nval, int ncol, double * params,
        double * inputs, double* innov)
{
	int i, j;
    double alpha, y0, value;

    /* Get AR1 coefficient */
    alpha = params[0];

    /* Loop across columns */
    for(j=0; j<ncol; j++)
    {
        y0 = params[1];

        /* loop through data */
        for(i=0; i<nval; i++)
        {
            value = inputs[ncol*i+j];
            innov[ncol*i+j] = value-alpha*y0;
            y0 = value;
        }
    }

    return 0;
}
