
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Maximum number of uh states*/
#define NUHMAXLENGTH 300

/* Check on uh sum */
#define UHEPS 0.0001

double uh_gr4j_ss1_daily(double ordinate, double lag);
double uh_gr4j_ss2_daily(double ordinate, double lag);
double uh_gr4j_ss1_hourly(double ordinate, double lag);
double uh_gr4j_ss2_hourly(double ordinate, double lag);
double uh_lag(double ordinate,double lag);
double uh_triangle(double ordinate,double lag);

double uh_delta(int uhid, double lag, double ordinate);

int c_uh_getnuhmaxlength(void);

double c_uh_getuheps(void);

int c_uh_getuh(int nuhlengthmax,
        int uhid, 
        double lag,
        int * nuh,
        double * uh);

int uh_runtimestep(int nuh, 
        double input, 
        double * uh, 
        double * states,
        double * outputs);

