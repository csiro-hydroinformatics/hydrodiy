
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_hymod_utils.h"

/* Number of inputs required by GR2M run */
#define GR2M_NINPUTS 5

/* Number of params required by GR2M run */
#define GR2M_NPARAMS 2

/* Number of states returned by GR2M run */
#define GR2M_NSTATES 5

/* Number of outputs returned by GR2M run */
#define GR2M_NOUTPUTS 20

int c_gr2m_getnstates(void);

int c_gr2m_getnoutputs(void);

int c_gr2m_getesize(void);

int c_gr2m_run(int nval, int nparams, int ninputs, 
        int nstates, int noutputs,
	double * params,
	double * inputs,
	double * statesini,
        double * outputs);

