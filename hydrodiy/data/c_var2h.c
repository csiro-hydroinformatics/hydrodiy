#include "c_var2h.h"


/*******************************************************************************
 * Routine permettant de convertir une variable a pas de temps variable
 * en variable a pas de temps fixe
 *
 * nvalvar : number of data points in variable time step series
 * nvalh : number of hourly time steps
 * display : display progress if >0
 * maxgapsec : Maximum number of seconds between missing data
 * varsec : Timestamp second for variable time step series
 * varvalues : Variable time step data
 * hstartsec : Starting second for hourly series
 * hvalues : hourly data
 *
 *****************************************************************************/
int c_var2h(int nvalvar, int nvalh, int display,
    int maxgapsec,
    int * varsec,
    double * varvalues,
    int hstartsec,
    double * hvalues)
{
    int ierr, i, varindex, miss;
    double a, hvalue, t1, t2, it1, it2, val1, val2;
    double start, end, nan, vali1, vali2;

    static double zero=0.;

    /* Display status */
    if(display==1)
    {
        fprintf(stdout, "\n\tConverting to fixed time step (%d variable ts values -> %d hourly values)..\n",
      		nvalvar, nvalh);
        fprintf(stdout, "\tMax gap between valid values : %d sec\n", maxgapsec);
        fprintf(stdout, "\tprogression (percent)..\n\t");
    }

    /* Set first time step to be immediately before hstart */
    varindex = 0;
    while(varsec[varindex]<=hstartsec) varindex++;
    varindex--;

    /* hstart is smaller than first value in varsec */
    if(varindex<0)
        return VAR2H_ERROR + __LINE__;

    /* Initialisation */
    nan = zero/zero;
    ierr = 0;

    /* Loop through instantaneous data */
    for(i=0; i<nvalh-1; i++)
    {
        if(display==1)
        {
            if(i%10000==0)
                fprintf(stdout, "%0.0f ",(double)i/(double)nvalh*100);
            if(i%50000==0 && i>0)
                fprintf(stdout, "\n\t");
        }

        /* Start and end of integration */
        start = (double)(hstartsec+i*3600);
        end = start+3600.;

        /* Initialisation */
        t1 = (double) varsec[varindex];
        val1 = varvalues[varindex];

        hvalue = 0;
        miss = 0;
        vali1 = 0.;
        vali2 = 0.;
        it1 = 0.;
        it2 = 0.;

        /* Calculate the hourly value */
        while(t1<end)
        {
            /* Get instantaneous time and values */
            t2 = (double) varsec[varindex+1];
            val2 = varvalues[varindex+1];

            if(t2<t1)
                return VAR2H_ERROR + __LINE__;

        	/*
            * prevent the calculation of hourly total
        	* Apply the maximum isolated criteria if both va1 and val2 are strictly positive
            */
            if(val1<-1e-8 || val2<-1e-8 || t2-t1>maxgapsec ||
                    isnan(val2) ||isnan(val1) )
            {
                miss=1;
                break;
            }

            /* Start and end of integration */
            it1 = t1<start ? start : t1;
            it2 = t2>end ? end : t2;

            /* Check that t1<t2, otherwise skip point */
            if(it2-it1>1e-8)
            {
                /* Compute interpolated values at start and end of
                 * interpolation period */
                a = (val2-val1)/(t2-t1);
                vali1 = a*(it1-t1)+val1;
                vali2 = a*(it2-t1)+val1;

                /* Trapezoidal integration */
                hvalue += (vali2+vali1)*(it2-it1)/2;
            }

            /* Loop */
            varindex++;
            t1 = t2;
            val1 = val2;

            /*
            fprintf(stdout, "[%d, %d] %0.1f %0.1f -> %0.1f %0.1f = %f\n",
                    i, varindex, it1, vali1, it2, vali2, hvalue);
            */
        }

        /* Store Hourly values */
        hvalues[i] = miss == 0 ? hvalue/3600. : nan;

        //fprintf(stdout, "hvalues[%d] = %f\n\n", i, hvalues[i]);
    }

    if(display==1) fprintf(stdout, "\n\n");

    return ierr;
}
