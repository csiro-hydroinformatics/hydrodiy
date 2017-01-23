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
int c_ar1innov(int nval, int ncol, double yini, double * alpha,
        double * innov, double* outputs)
{
	int i, j;
    double value, y0, y1, nan;

    /* nan value if not defined */
    nan = 0./0.;

    /* loop across columns */
    for(j=0; j<ncol; j++)
    {
        /* Initialise AR1 */
        y0 = yini;
        y1 = y0;

        /* loop through values */
        for(i=0; i<nval; i++)
        {
            value = innov[ncol*i+j];

            /* Process nan values */
            if(isnan(value))
            {
                /* Set output to nan */
                y1 = nan;

                /* Decrease states towards zero */
                y0 *= alpha[i];
            }
            else
            {
                y0 = alpha[i]*y0 + value;
                y1 = y0;
            }

            outputs[ncol*i+j] = y1;
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
int c_ar1inverse(int nval, int ncol, double yini, double * alpha,
        double * inputs, double* innov)
{
	int i, j;
    double y0, y1, nan, value;

    /* nan value if not defined */
    nan = 0./0.;

    /* Loop across columns */
    for(j=0; j<ncol; j++)
    {
        y0 = yini;

        /* loop through data */
        for(i=0; i<nval; i++)
        {
            value = inputs[ncol*i+j];

            /* Process nan values */
            if(isnan(value))
            {
                /* Set output to nan */
                y1 = nan;

                /* Decrease states towards zero */
                y0 *= alpha[i];
            }
            else
            {
                y1 = value-alpha[i]*y0;
                y0 = value;
            }

            innov[ncol*i+j] = y1;
        }
    }

    return 0;
}
