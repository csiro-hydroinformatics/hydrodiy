#include "c_gr4j.h"


/*** GR4J Model *********************************************
* Code written by Julien Lerat, Bureau of Meteorology
*
	nstates = 11

	nparams = 4
	params[0] : S
	params[1] : IGF
	params[2] : R
	params[3] : TB
*/

int c_gr4j_getnstates(void)
{
    return GR4J_NSTATES;
}

int c_gr4j_getnuh(void)
{
    return GR4J_NUH;
}

int c_gr4j_getnoutputs(void)
{
    return GR4J_NOUTPUTS;
}


double SS1(double I,double C)
{
    double s = I<0 ? 0 :
        I<C ? pow(I/C, GR4J_UHEXPON) : 1;

    return s;
}

double SS2(double I,double C)
{
    double s = I<0 ? 0 :
        I<C ? 0.5*pow(I/C, GR4J_UHEXPON) :
        I<2*C ? 1-0.5*pow(2-I/C, GR4J_UHEXPON) : 1;

    return s;
}

int gr4j_minmaxparams(int nparams, double * params)
{
        if(nparams<4)
            return EINVAL;

	params[0] = c_hymod_minmax(1,1e5,params[0]); 	// S
	params[1] = c_hymod_minmax(-50,50,params[1]);	// IGF
	params[2] = c_hymod_minmax(1,1e5,params[2]); 	// R
	params[3] = c_hymod_minmax(0.5,50,params[3]); // TB

	return 0;
}

/*******************************************************************************
* Initialise gr4j uh and states
*
*
*******************************************************************************/
int c_gr4j_getuh(double lag,
        int * nuh_optimised,
        double * uh)
{
	int i, nuh1;
        double Sa, Sb;

        lag = lag < 0 ? 0 : lag;

	/* UH ordinates */
        nuh1 = 0;
	for(i=0; i<GR4J_NUH-1; i++)
        {
	    Sb = SS1((double)(i+1), lag);
            Sa = SS1((double)(i), lag);
            uh[i] = Sb-Sa;

            if(1-Sb < GR4J_UHEPS)
            {
                nuh1 = i+1;
                /* ideally we should correct uh here
                but I know GR4J UH is accurate */
                break;
            }
        }

        /* NUH is not big enough */
        if(1-Sb > GR4J_UHEPS || nuh1 > (GR4J_NUH-1)/3)
        {
            fprintf(stderr, "%s:%d:ERROR: GR4J_NUH(%d) is not big enough\n",
                __FILE__, __LINE__, GR4J_NUH);
            return EINVAL;
        }

	for(i=0; i < 2*nuh1; i++)
        {
	    Sb = SS2((double)(i+1), lag);
            Sa = SS2((double)(i), lag);
            uh[nuh1 + i] = Sb-Sa;
        }

        *nuh_optimised = 3*nuh1;

	return 0;
}

/*******************************************************************************
* Run time step code for the GR4J rainfall-runoff model
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
*					params[2] = R
*					params[3] = TB
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

int c_gr4j_runtimestep(int nparams, int nuh, int ninputs,
        int nstates, int noutputs,
	double * params,
        double * uh,
        double * inputs,
	double * statesuh,
        double * states,
        double * outputs)
{
        int ierr=0, k,l, nuh1, nuh2;

	double Q, P, E;
	double ES, PS, PR, WS,S2;
        double PERC,ECH,TP,R2,QR,QD;
	double EN, ech1,ech2;

	/* UH dimensions */
	nuh1 = nuh/3;
	nuh2 = 2*nuh/3;

	/* inputs */
	P = inputs[0] < 0 ? 0 : inputs[0];
	E = inputs[1] < 0 ? 0 : inputs[1];

        states[0] = c_hymod_minmax(0, params[0], states[0]);
        states[1] = states[1] < 0 ? 0 : states[1];

	/* Production */
	if(P>E)
	{
		WS = (P-E)/params[0];
        WS = WS > 13 ? 13 : WS;

		PS = params[0]*(1-pow(states[0]/params[0],2))*tanh(WS);
        PS /= (1+states[0]/params[0]*tanh(WS));

		ES = 0;
		PR = P-E-PS;
		EN = 0;
	}
	else
	{
		WS =(E-P)/params[0];
        WS = WS > 13 ? 13 : WS;

		ES = states[0]*(2-states[0]/params[0])*tanh(WS);
        ES /= (1+(1-states[0]/params[0])*tanh(WS));

		PS = 0;
		PR = 0;
		EN = E-P;
	}

	states[0] += PS-ES;

	/* Percolation */
	S2 = states[0]/pow(1+pow(states[0]/GR4J_PERCFACTOR/params[0],4),0.25);

	PERC = states[0]-S2;
	states[0] = S2;

	PR += PERC;

	/* UH1 */
	for (k=0;k<nuh1-1;k++)
            statesuh[k] = statesuh[1+k]+uh[k]*PR;

	statesuh[nuh1-1] = uh[nuh1-1]*PR;

	/* UH2 */
	for (l=0;l<nuh2-1;l++)
            statesuh[nuh1+l] = statesuh[nuh1+1+l]+uh[nuh1+l]*PR;

	statesuh[(nuh1+nuh2)-1] = uh[(nuh1+nuh2)-1]*PR;

	/* Potential Water exchange
	ECH=XV(NPX+3)*(X(1)/XV(NPX+1))**3.5  // Formulation initiale
	ECH=XV(NPX+3)*(X(1)/XV(NPX+1)-XV(NPX+5)) // Formulation N. Lemoine
        */
	ECH = params[1]*pow(states[1]/params[2],3.5);

	/* Routing store calculation */
	TP = states[1]+statesuh[0]*0.9+ECH;

	/* Case where Reservoir content is not sufficient */
	ech1 = ECH-TP;
    states[1] = 0;

	if(TP>=0)
    {
        states[1]=TP;
        ech1=ECH;
    }

	R2 = states[1]/pow(1+pow(states[1]/params[2],4),0.25);
	QR = states[1]-R2;
	states[1] = R2;

	/* Direct runoff calculation */
	QD = 0;

	/* Case where the UH cannot provide enough water */
	TP = statesuh[nuh1]*0.1+ECH;
	ech2 = ECH-TP;
        QD=0;

	if(TP>0)
    {
        QD=TP;
        ech2=ECH;
    }

	/* TOTAL STREAMFLOW */
	Q = QD+QR;

	/* RESULTS */
	outputs[0] = Q;

    if(noutputs>1)
	    outputs[1] = ech1+ech2;
	else
		return ierr;

    if(noutputs>2)
	    outputs[2] = ES+EN;
	else
		return ierr;

    if(noutputs>3)
	    outputs[3] = PR;
	else
		return ierr;

   	if(noutputs>4)
	    outputs[4] = QD;
	else
		return ierr;

    if(noutputs>5)
	    outputs[5] = QR;
	else
		return ierr;

    if(noutputs>6)
	    outputs[6] = PERC;
	else
		return ierr;

    if(noutputs>7)
	    outputs[7] = states[0];
	else
		return ierr;

    if(noutputs>8)
	    outputs[8] = states[1];
	else
		return ierr;

	return ierr;
}


// --------- Component runner --------------------------------------------------
int c_gr4j_run(int nval, int nparams, int nuh, int ninputs,
        int nstates, int noutputs,
	double * params,
        double * uh,
	double * inputs,
        double * statesuhini,
	double * statesini,
        double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(nparams < 4)
        return HYMOD_ESIZE;

    if(nstates < 2)
        return HYMOD_ESIZE;

    if(ninputs < 2)
        return HYMOD_ESIZE;

    if(noutputs > GR4J_NOUTPUTS)
        return HYMOD_ESIZE;

    if(nuh > GR4J_NUH)
        return HYMOD_ESIZE;

    /* Check parameters */
    ierr = gr4j_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_gr4j_runtimestep(nparams, nuh, ninputs,
                nstates, noutputs,
    		params,
                uh,
                &(inputs[ninputs*i]),
                statesuhini,
                statesini,
                &(outputs[noutputs*i]));
    }

    return ierr;
}

