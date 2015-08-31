#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


/* ************** Core subroutine ******************************
* Compute the baseflow computation based on the 3 algorithms described 
* by 
* Chapman, T. (1999), A comparison of algorithms for stream flow recession and baseflow separation. 
* Hydrol. Process., 13: 701–714. doi: 10.1002/(SICI)1099-1085(19990415)13:5<701::AID-HYP774>3.0.CO;2-2
*
* Doc:
* method = Baseflow separation method (1=1 parameter, 2=Boughton, 3=IHACRES)
* params = algorithm parameters
* nval = length of input vectors (number of values)
* inputs = flow time series
* outputs = baseflow series
*
************************************************/
int c_baseflow(int method, int nval, 
	double* params, 
	double* inputs, 
	double* outputs)
{
	int i;
	double q, qp, bf, bfp,  k=0, C=0, a=0;

    /* Check params */
    k = params[0];
    if(k<0 || k>1)
        return EDOM;

    if(method >= 2)
    {
		C = params[1];
		if(C<0)
        	return EDOM;
    }

    if(method >= 3)
    {
		a = params[2];
		if(a<0 || a>1)
        	return EDOM;
    }

    /* Initisalise */
    q = inputs[0];
    if(q<0) q=0;
    qp = q;
    bf = q;
    bfp = q;
    outputs[0] = bf;

    /* loop through data */
    for(i=1; i<nval; i++)
    {
        q = inputs[i]>=0 ? inputs[i] : q;

	/* One parameter - Chapman */
        if (method == 1)
            bf = k*bfp/(2-k) + (1-k)*q/(2-k);

	/* Two parameters - Boughton */
        if (method == 2)
            bf = k*bfp/(1+C) + C*q/(1+C);

	/* Three parameters - IHACRES */
        if(method == 3)
            bf = k*bfp/(1+C) + C*(q+a*qp)/(1+C);

        /* Discard floods */
        if (bf>q)
            bf = q;

	/* Loop */
	qp ++;
	qp = q;

	bfp++;
	bfp = bf;
	outputs[i] = bf;
    }

    return 0;
}
