#include "c_armodels.h"

/* nan value if not defined */
double get_nan(void)
{
    double nan;
    static double zero = 0.0;

    nan = 1./zero * zero;
    return nan;
}

/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = params x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 paramter
*   params[1] : initial value of ar1 process
* prev = initialisation
* innov = innovation series e[i]
* outputs = model output
*
************************************************/
int c_armodel_sim(int nval, int ncols, int nparams,
        int fillnan,
        double simini, double * params,
        double * innov, double* outputs)
{
	int i, j, k;
    double value, tmp;
    double prev[ARMODEL_NPARAMSMAX];

    /* Check AR order */
    if(nparams > ARMODEL_NPARAMSMAX)
        return ARMODEL_ERROR+__LINE__;

    /* Check parameter values */
    for(k=0; k<nparams; k++)
        if(isnan(params[k]))
            return ARMODEL_ERROR+__LINE__;

    /* Check initial condition */
    if(isnan(simini))
        return ARMODEL_ERROR+__LINE__;

    /* loop across columns */
    fprintf(stdout, "fillnan = %d\n", fillnan);
    for(j=0; j<ncols; j++)
    {
        /* Initialise AR model */
        for(k=0; k<nparams; k++)
            prev[k] = simini;

        /* loop through values */
        for(i=0; i<nval; i++)
        {
            value = innov[ncols*i+j];

            /* Run AR model */
            tmp = value;
            for(k=nparams-1; k>=0; k--)
            {
                /* Check nan value */
                if(isnan(prev[k]))
                    prev[k] = simini;

                /* Run AR model  */
                tmp += params[k]*prev[k];

                fprintf(stdout, "p[%2d,%2d]=%5.2f ", i, k, prev[k]);

                /* Loop prev vector */
                if(isnan(tmp))
                    tmp = fillnan > 0 ? simini : 0;
                prev[k] = k>0 ? prev[k-1] : tmp;
            }
            fprintf(stdout, " | %5.2f -> %5.2f\n", value, tmp);

            /* Store simulated value */
            outputs[ncols*i+j] = tmp;
        }
    }

    return 0;
}

/* ************** Core subroutine ******************************
* Computes innovation from ar series :
*   y[i+1] = Sum(params[k] x y[i-k], k=1, n) + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 parameter
* inputs = ar1 timeseries
* residual = innovation series e[i]
*
************************************************/
int c_armodel_residual(int nval, int ncols, int nparams,
        int fillnan,
        double stateini, double * params,
        double * inputs, double* residuals)
{
	int i, j, k;
    double value, tmp;
    double prev[ARMODEL_NPARAMSMAX];

    /* Check AR order */
    if(nparams > ARMODEL_NPARAMSMAX)
        return ARMODEL_ERROR+__LINE__;

    /* Check parameter values */
    for(k=0; k<nparams; k++)
        if(isnan(params[k]))
            return ARMODEL_ERROR+__LINE__;

    /* Check initial condition */
    if(isnan(stateini))
        return ARMODEL_ERROR+__LINE__;

    /* Loop across columns */
    for(j=0; j<ncols; j++)
    {
        /* Initialise AR model */
        for(k=0; k<nparams; k++)
            prev[k] = stateini;

        /* loop through data */
        for(i=0; i<nval; i++)
        {
            value = inputs[ncols*i+j];

            /* When value is nan, estimate it using AR model */
            if(isnan(value) && fillnan > 0)
            {
                value = 0;
                for(k=0; k<nparams; k++)
                    value += params[k]*prev[k];
            }

            /* Compute AR1 residual */
            tmp = value;
            for(k=nparams-1; k>=0; k--)
            {
                /* Run AR model */
                tmp -= params[k]*prev[k];

                /* Loop prev vector */
                prev[k] = k>0 ? prev[k-1] : tmp;
            }

            /* Loop */
            residuals[ncols*i+j] = tmp;
        }
    }

    return 0;
}
