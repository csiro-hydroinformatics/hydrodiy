#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "c_lagroute.h"
#include "c_uh.h"
#include "c_utils.h"


int c_lagroute_getnstates(void)
{
    return LAGROUTE_NSTATES;
}

int c_lagroute_getnoutputs(void)
{
    return LAGROUTE_NOUTPUTS;
}


int lagroute_minmaxparams(int nparams, double * params)
{
        double p1, p2;

        if(nparams<2)
            return EMODEL_RUN;

        p1 = params[0];
        params[0] = p1 < 1e-2 ? 1e-2 :
            p1 > 20 ? 20 : p1;

        p2 = params[1];
        params[1] = p2 < 0. ? 0. :
            p2 > 1. ? 1. : p2;

	return 0;
}


int c_lagroute_runtimestep(int nparams,
        int nuh, int ninputs,
        int nstates, int noutputs,
        double dt, double L, double qstar, int storage_type,
	double * params,
        double * uh,
        double * inputs,
	double * statesuh,
        double * states,
        double * outputs)
{
    int k, ierr = 0;
    double q1, q1lag;
    double v0, v1, vr, U, theta1, omega;
    double alpha;
    double vtanh;

    /* input */
    q1 = inputs[0];
    q1 = q1 < 0 ? q1 : q1;

    /* states */
    v0 = states[0];
    v0 = v0 != v0 ? 0 : v0;

    /* Reparameterise */
    U = params[0];
    alpha = params[1];
    theta1 = (1-alpha) * qstar * L * U;

    /* Lag component */
    vr = 0;
    for (k=0;k<nuh-1;k++)
    {
        statesuh[k] = statesuh[1+k]+uh[k]*q1;

        /* Volume in transit in the river reach */
        if(k>0)
            vr += statesuh[k]*dt;
    }
    statesuh[nuh-1] = uh[nuh-1]*q1;

    if(nuh>1)
        vr += statesuh[nuh-1]*dt;

    q1lag = statesuh[0];

    /* Storage component */

    v1 = v0;

    if(theta1>0)
    {
        if(storage_type == 1)
        {
            if(q1lag>0)
            {
                omega = theta1 * q1lag/qstar;
                v1 = omega * (1-(1-v0/omega) * exp(-q1lag*dt/omega));
            }
            else
                v1 = v0 * exp(-qstar*dt/theta1);
        }
        else if(storage_type == 2)
        {

            if(q1lag>0)
            {
                omega = theta1 * sqrt(q1lag/qstar);
                vtanh = tanh(omega*dt*q1lag);
                v1 = (v0+vtanh/omega)/(1+omega*v0*vtanh);
            }
            else
                v1 = v0/(1+v0/theta1/theta1*qstar*dt);
        }
        else
            return EMODEL_RUN;
    }

    /* States */
    states[0] = v1;

    /* flow outputs */
    outputs[0] = q1lag - (v1-v0)/dt;

    if(noutputs > 1)
        outputs[1] = q1lag;
    else
        return ierr;

    /* Storage outputs */
    if(noutputs > 2)
        outputs[2] = vr;
    else
        return ierr;

    if(noutputs > 3)
        outputs[3] = v1;
    else
        return ierr;

    return ierr;
}

// --------- Component runner --------------------------------------------------
int c_lagroute_run(int nval,
        int nparams,
        int nuh,
        int ninputs,
        int nconfig,
        int nstates,
        int noutputs,
        double * config,
	double * params,
        double * uh,
	double * inputs,
        double * statesuh,
	double * states,
        double * outputs)
{
    int ierr=0, i, storage_type;
    double dt, L, qstar, theta2;

    /* Check dimensions */
    if(nparams < 2)
        return ESIZE_PARAMS;

    if(nconfig < 4)
        return ESIZE_CONFIG;

    if(nstates < 1)
        return ESIZE_STATES;

    if(nuh > NUHMAXLENGTH)
        return ESIZE_STATESUH;

    if(noutputs > LAGROUTE_NOUTPUTS)
        return ESIZE_OUTPUTS;


    /* Config data */
    dt = config[0];
    dt = dt < 1 ? 1 : dt;

    L = config[1];
    L = L < 1 ? 1 : L;

    qstar = config[2];
    qstar = qstar < 1e-5 ? 1e-5 : qstar;

    theta2 = config[3];
    if(fabs(theta2-1) < 1e-20)
    {
        storage_type = 1;
    }
    else if(fabs(theta2-2) < 1e-20)
    {
        storage_type = 2;
    }
    else
    {
        fprintf(stderr, "%s:%d:ERROR: theta2(%f) is neither 1 nor 2\n",
            __FILE__, __LINE__, theta2);
        return EMODEL_RUN;
    }

    /* Check parameters */
    ierr = lagroute_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_lagroute_runtimestep(nparams, nuh, ninputs,
                nstates, noutputs,
                dt, L, qstar, storage_type,
    		params,
                uh,
                &(inputs[ninputs*i]),
                statesuh,
                states,
                &(outputs[noutputs*i]));
    }

    return ierr;
}

