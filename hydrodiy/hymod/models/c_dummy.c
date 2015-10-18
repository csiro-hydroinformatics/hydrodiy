#include "c_dummy.h"
#include "c_uh.h"

int c_dummy_getnstates(void)
{
    return DUMMY_NSTATES;
}


int c_dummy_getnoutputs(void)
{
    return DUMMY_NOUTPUTS;
}


int dummy_runtimestep(int nparams,
    int ninputs,
    int nstates,
    int noutputs,
	double * params,
    double * inputs,
	double * states,
    double * outputs)
{
    outputs[0] = params[0] * inputs[0];
    states[0] += inputs[0];

    if(noutputs > 1)
        outputs[1] = inputs[1];

    return 0;
}


// --------- Component runner --------------------------------------------------
int c_dummy_run(int nval,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
	double * params,
	double * inputs,
	double * statesini,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(noutputs > DUMMY_NOUTPUTS)
        return ESIZE_OUTPUTS;

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
        /* Run timestep model and update states */
    	ierr = dummy_runtimestep(nparams,
                ninputs,
                nstates, noutputs,
    		    params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));
    }

    return ierr;
}

