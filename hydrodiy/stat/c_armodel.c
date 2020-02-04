#include "c_armodel.h"

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
* sim = initialisation
* innov = innovation series e[i]
* outputs = model output
*
************************************************/
int c_armodel_sim(int nval, int ncol, int nparams,
        double simini, double * params,
        double * innov, double* outputs)
{
	int i, j, k;
    double value;
    double sim[ARMODEL_NPARAMSMAX];

    /* More than 1 AR parameters not implemented for now */
    if(nparams > 1)
        return ARMODEL_ERROR+__LINE__;

    /* loop across columns */
    for(j=0; j<ncol; j++)
    {
        /* Initialise AR model */
        for(k=0; k<nparams; k++)
            sim[k] = simini;

        /* loop through values */
        for(i=0; i<nval; i++)
        {
            value = innov[ncol*i+j];

            /* Process nan values */
            if(isnan(value))
            {
                /* No an AR simul to estimate current value */
                sim[0] *= params[0];

                /* Store nan */
                outputs[ncol*i+j] = get_nan();
            }
            else
            {
                /* Implement AR1 */
                sim[0] = params[0]*sim[0] + value;

                /* Store simulated value */
                outputs[ncol*i+j] = sim[0];
            }

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
* inputs = ar1 timeseries
* residual = innovation series e[i]
*
************************************************/
int c_armodel_residual(int nval, int ncol, int nparams,
        double stateini, double * params,
        double * inputs, double* residuals)
{
	int i, j, k;
    double value;
    double prev[ARMODEL_NPARAMSMAX];

    /* More than 1 AR parameters not implemented for now */
    if(nparams > 1)
        return ARMODEL_ERROR+__LINE__;

    fprintf(stdout, "nv=%d nc=%d np=%d\n", nval, ncol, nparams);

    /* Loop across columns */
    for(j=0; j<ncol; j++)
    {
        /* Initialise AR model */
        for(k=0; k<nparams; k++)
            prev[k] = stateini;

        /* loop through data */
        for(i=0; i<nval; i++)
        {
            value = inputs[ncol*i+j];

            /* Process nan values */
            if(isnan(value))
            {
                /* Run AR simulation to estimate current value */
                prev[0] *= params[0];

                /* Store nan */
                residuals[ncol*i+j] = get_nan();
            }
            else
            {
                /* Compute AR1 residual */
                residuals[ncol*i+j] = value-params[0]*prev[0];

                /* Loop */
                prev[0] = value;
            }

        }
    }

    return 0;
}
