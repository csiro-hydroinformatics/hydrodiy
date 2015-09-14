#include "c_gr2m.h"
#include "c_hymod_utils.h"


/*** GR2M Model *********************************************
* Code written by Julien Lerat, Bureau of Meteorology
*
	nstates = 11

	nparams = 4
	params[0] : S
	params[1] : IGF
*/

int c_gr2m_getnstates(void)
{
    return GR2M_NSTATES;
}

int c_gr2m_getnoutputs(void)
{
    return GR2M_NOUTPUTS;
}

int gr2m_minmaxparams(int nparams, double * params)
{
        if(nparams<4)
            return EINVAL;

	params[0] = c_hymod_minmax(1,1e5,params[0]); 	// S
	params[1] = c_hymod_minmax(0,3,params[1]);	// IGF

	return 0;
}


/*******************************************************************************
* Run time step code for the GR2M rainfall-runoff model
* 
* --- Inputs
* ierr			Error message
* nconfig		Number of configuration elements (1)
* nparams			Number of paramsameters (4)
* ninputs		Number of inputs (2)
* nstates		Number of states (1 output + 2 model states + 8 variables = 11)
* nuh			Number of uh ordinates (2 uh)
*
* params			Model paramsameters. 1D Array nparams(4)x1
*					params[0] = S
*					params[1] = IGF
*
* uh			uh ordinates. 1D Array nuhx1
*
* inputs		Model inputs. 1D Array ninputs(2)x1
*
* statesuh		uh content. 1D Array nuhx1
*
* states		Output and states variables. 1D Array nstates(11)x1
*
*******************************************************************************/
 
int c_gr2m_runtimestep(int nparams, int ninputs, 
        int nstates, int noutputs,
	double * params,
        double * inputs,
        double * states,
        double * outputs)
{
        int ierr=0;
        
        /* parameters */
        double Scapacity = params[0];
        double IGFcoef = params[1];
        double Rcapacity = 60;

        /* model variables */
        double P, E;
        double S, R, S1, S2, PHI, PSI, P1, P2, P3;
        double R1, R2, F, Q;

    	/* inputs */
	P = inputs[0] < 0 ? 0 : inputs[0];
	E = inputs[1] < 0 ? 0 : inputs[1];

        S = c_hymod_minmax(0, params[0], states[0]);
        R = states[1] < 0 ? 0 : states[1];

        /* main GR2M procedure */

        /* production */
        PHI = tanh(P/Scapacity);
        S1 = (S+Scapacity*PHI)/(1+PHI*S/Scapacity);
        P1 = P+S-S1;

        PSI = tanh(E/Scapacity);
        S2 = S1*(1-PSI)/(1+PSI*(1-S1/Scapacity)); 

        S = S2/pow(1+pow(S2/Scapacity, 3), 1./3);
        P2 = S2-S;
        P3 = P1 + P2;

        /* routing */
        R1 = R + P3; 
        R2 = IGFcoef * R1; 
        F = (IGFcoef-1)*R1;
        Q = pow(R2,2)/(R2+Rcapacity);
        R = R2-Q;

        /* states */
        states[0] = S;
        states[1] = R;

        /* output */
        outputs[0] = Q;

        if(noutputs>1)
            outputs[1] = F;

        if(noutputs>2)
            outputs[2] = P1;

        if(noutputs>3)
            outputs[3] = P2;

        if(noutputs>4)
            outputs[4] = P3;

        if(noutputs>5)
            outputs[5] = R1;
        
        if(noutputs>6)
            outputs[6] = R2;

        if(noutputs>7)
            outputs[7] = S/Scapacity;

        if(noutputs>8)
            outputs[8] = R/Rcapacity;

	return ierr;
}


// --------- Component runner --------------------------------------------------
int c_gr2m_run(int nval, int nparams, int ninputs, 
        int nstates, int noutputs, 
	double * params,
	double * inputs,
	double * statesini,
        double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(nparams < 2)
        return HYMOD_ESIZE;

    if(nstates < 2)
        return HYMOD_ESIZE;

    if(ninputs < 2)
        return HYMOD_ESIZE;

    if(noutputs > 9)
        return HYMOD_ESIZE;

    /* Check parameters */
    ierr = gr2m_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_gr2m_runtimestep(nparams, ninputs, 
                nstates, noutputs,
    		params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));
    }
    
    return ierr;
}

