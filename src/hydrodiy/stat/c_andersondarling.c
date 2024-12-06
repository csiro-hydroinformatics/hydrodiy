#include "c_andersondarling.h"

/* comparison function for qsort ***/
static int compare(const void* p1, const void* p2)
{
    double a,b;
    a = *(double *)p1;
    b = *(double *)p2;
    if(a>b) return 1;
    if(a==b) return 0;
    if(a<b) return -1;
    return 0;
}

/* The 3 following functions are not used at the moment */
int c_ad_probexactinf(int nval, double *unifdata, double *prob)
{
    int i;
    for(i = 0; i < nval; i++)
      prob[i] = ADinf(unifdata[i]);

    return 0;
}

int c_ad_probn(int nval, int nsample, double *unifdata, double *prob)
{
    int i;
    for(i = 0; i < nval; i++)
      prob[i] = AD(nsample, unifdata[i]);

    return 0;
}

int c_ad_probapproxinf(int nval, double *unifdata, double *prob)
{
    int i;
    for(i = 0; i < nval; i++)
      prob[i] = adinf(unifdata[i]);

    return 0;
}

/* Main function to run the Anderson-Darling test */
int c_ad_test(int nval, double *unifdata, double *outputs)
{
    /* sort unifdata */
    qsort(unifdata, nval, sizeof(double), compare);

    /* Compute AD statistic and p-value and
    * store them in the outputs vector */
    return ADtest(nval, unifdata, outputs);
}
