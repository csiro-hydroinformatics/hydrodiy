#include "c_armodels.h"

/*
 * Code to run auto-regressive models.
 *
 * !! CAUTION, this code does not check model
 * parameter stability. It may lead to explosive
 * behaviour.
 *
 * */

/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = sum(params[k] x y[i-k], k=0, n) + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 paramter
*   params[1] : initial value of ar1 process
* sim_mean = Mean shift
* sim_ini = initialisation
* innov = innovation series e[i]
* outputs = model output
*
************************************************/
int c_armodel_sim(int nval, int nparams,
        double sim_mean,
        double sim_ini, double * params,
        double * innov, double* outputs)
{
	int i, k;
    double value, tmp;
    double prev_centered[ARMODEL_NPARAMSMAX];

    /* Check AR order */
    if(nparams > ARMODEL_NPARAMSMAX || nparams <= 0)
        return ARMODEL_ERROR+__LINE__;

    /* Check parameter values */
    for(k=0; k<nparams; k++)
        if(isnan(params[k]))
            return ARMODEL_ERROR+__LINE__;

    /* Check mean and initial condition */
    if(isnan(sim_mean))
        return ARMODEL_ERROR+__LINE__;

    if(isnan(sim_ini))
        return ARMODEL_ERROR+__LINE__;

    /* Initialise AR model */
    for(k=0; k<nparams; k++)
        prev_centered[k] = (sim_ini-sim_mean);

    /* loop through values */
    for(i=0; i<nval; i++)
    {
        value = innov[i];

        /* Check nan innov, replace by zero */
        value = isnan(value) ? 0 : value;

        /* Run AR model */
        tmp = value;
        for(k=nparams-1; k>=0; k--)
        {
            /* Run AR model  */
            if(!isnan(prev_centered[k]))
                tmp += params[k]*prev_centered[k];

            /* Loop prev_centered vector */
            prev_centered[k] = k>0 ? prev_centered[k-1] : tmp;
        }

        /* Store simulated value */
        outputs[i] = tmp+sim_mean;
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
int c_armodel_residual(int nval, int nparams,
        double sim_mean,
        double sim_ini, double * params,
        double * inputs, double* residuals)
{
	int i, k;
    double value, tmp;
    double prev_centered[ARMODEL_NPARAMSMAX];

    /* Check AR order */
    if(nparams > ARMODEL_NPARAMSMAX || nparams <= 0)
        return ARMODEL_ERROR+__LINE__;

    /* Check parameter values */
    for(k=0; k<nparams; k++)
        if(isnan(params[k]))
            return ARMODEL_ERROR+__LINE__;

    /* Check mean and initial condition */
    if(isnan(sim_mean))
        return ARMODEL_ERROR+__LINE__;

    if(isnan(sim_ini))
        return ARMODEL_ERROR+__LINE__;

    /* Initialise AR model */
    for(k=0; k<nparams; k++)
        prev_centered[k] = sim_ini-sim_mean;

    /* loop through data */
    for(i=0; i<nval; i++)
    {
        value = inputs[i]-sim_mean;

        /* When value is nan, estimate it using AR model
         * converging to sim_mean. This will lead to
         * 0 residual for indexes corresponding to
         * nan values in inputs. However, it will influence
         * the residual computation for the following time
         * steps if order > 1.
         * */
        if(isnan(value))
        {
            value = 0;
            for(k=0; k<nparams; k++)
                value += params[k]*prev_centered[k];
        }

        /* Compute AR residual */
        tmp = value;
        for(k=nparams-1; k>=0; k--)
        {
            /* Run AR model */
            tmp -= params[k]*prev_centered[k];

            /* Loop prev_centered vector */
            prev_centered[k] = k>0 ?
                    prev_centered[k-1] : value;
        }

        /* Loop */
        residuals[i] = tmp;
    }

    return 0;
}
