#include "c_gr4j.h"
#include "c_uh.h"

int c_gr4j_getnstates(void)
{
    return GR4J_NSTATES;
}


int c_gr4j_getnoutputs(void)
{
    return GR4J_NOUTPUTS;
}

int gr4j_minmaxparams(int nparams, double * params)
{
        if(nparams<4)
            return EINVAL;

	params[0] = c_utils_minmax(1,1e5,params[0]); 	// S
	params[1] = c_utils_minmax(-50,50,params[1]);	// IGF
	params[2] = c_utils_minmax(1,1e5,params[2]); 	// R
	params[3] = c_utils_minmax(0.5,50,params[3]); // TB

	return 0;
}

int c_gr4j_production(double P, double E, 
        double Scapacity, 
        double S, 
        double * prod)
{
    double WS, PS, ES, EN=0, PR, PERC, S2;

	/* production store */
	if(P>E)	
	{
		WS =(P-E)/Scapacity;
        WS = WS >= 13 ? 13 : WS;

		ES = 0;
		PS = Scapacity*(1-pow(S/Scapacity,2))*tanh(WS);
        PS /= (1+S/Scapacity*tanh(WS));
		PR = P-E-PS;
		EN = 0;
	}
	else	
	{
		WS = (E-P)/Scapacity;
        WS = WS >= 13 ? 13 : WS;

		ES = S*(2-S/Scapacity)*tanh(WS);
        ES /= (1+(1-S/Scapacity)*tanh(WS));
		PS = 0;
		PR = 0;
		EN = E-P;
	}
	S += PS-ES;

	/* percolation */
	S2 = S/pow(1+pow(S/GR4J_PERCFACTOR/Scapacity,4),0.25);
	PERC = S-S2;
	S = S2;
	PR += PERC;

    prod[0] = EN;
    prod[1] = PS;
    prod[2] = ES;
    prod[3] = PERC;
    prod[4] = PR;
	prod[5] = S;

    return 0;
}


int c_gr4j_runtimestep(int nparams, 
    int nuh1, int nuh2, int ninputs,
    int nstates, int noutputs,
	double * params,
    double * uh1,
    double * uh2,
    double * inputs,
	double * statesuh,
    double * states,
    double * outputs)
{
    int ierr=0;

	double Q, P, E;
    double prod[6];
	double ES, PS, PR;
    double PERC,ECH,TP,R2,QR,QD;
	double EN, ech1,ech2;
    double uhoutput1[1], uhoutput2[1];

	/* inputs */
	P = inputs[0];
    P = P < 0 ? 0 : P;

	E = inputs[1];
    E = E < 0 ? 0 : E; 

    /* Production */
    c_gr4j_production(P, E, params[0], states[0], prod);

    EN = prod[0];
    PS = prod[1];
    ES = prod[2];
    PERC = prod[3];
    PR = prod[4];
    states[0] = prod[5];

	/* UH */
    c_uh_runtimestep(nuh1, PR, uh1, statesuh, uhoutput1); 
    c_uh_runtimestep(nuh2, PR, uh2, &(statesuh[nuh1]), uhoutput2); 

	/* Potential Water exchange
	ECH=XV(NPX+3)*(X(1)/XV(NPX+1))**3.5  // Formulation initiale
	ECH=XV(NPX+3)*(X(1)/XV(NPX+1)-XV(NPX+5)) // Formulation N. Lemoine
        */
	ECH = params[1]*pow(states[1]/params[2],3.5);

	/* Routing store calculation */
	TP = states[1] + *uhoutput1 * 0.9 + ECH;

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
	TP = *uhoutput2 * 0.1 + ECH;
	ech2 = ECH-TP;
    QD = 0;

	if(TP>0)
    {
        QD = TP;
        ech2 = ECH;
    }

	/* TOTAL STREAMFLOW */
	Q = QD + QR;

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
int c_gr4j_run(int nval, int nparams, 
    int nuh1, int nuh2, int ninputs,
    int nstates, int noutputs,
	double * params,
    double * uh1,
    double * uh2,
	double * inputs,
    double * statesuhini,
	double * statesini,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(nparams < 4)
        return MODEL_ESIZE;

    if(nstates < 2)
        return MODEL_ESIZE;

    if(ninputs < 2)
        return MODEL_ESIZE;

    if(noutputs > GR4J_NOUTPUTS)
        return MODEL_ESIZE;

    if(nuh1 > NUHMAXLENGTH)
        return MODEL_ESIZE;

    if(nuh2 > NUHMAXLENGTH)
        return MODEL_ESIZE;

    /* Check parameters */
    ierr = gr4j_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
        /* Run timestep model and update states */
    	ierr = c_gr4j_runtimestep(nparams, 
                nuh1, nuh2, ninputs,
                nstates, noutputs,
    		    params,
                uh1, uh2,
                &(inputs[ninputs*i]),
                statesuhini,
                statesini,
                &(outputs[noutputs*i]));
    }

    return ierr;
}

