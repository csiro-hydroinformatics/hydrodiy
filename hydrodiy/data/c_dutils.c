#include "c_dutils.h"

/**
* Aggregate inputs based on the aggindex
* The operator applied to the aggregation is defined by op:
* op = 0 : sum
* op = 1 : mean
**/
int c_aggregate(int nval, int oper, int * aggindex,
    double * inputs, double * outputs, int * iend)
{
    int i, ncount, count, ia, iaprev;
    double sum;

    iaprev = aggindex[0];
    ia = 0;
    count = 0;
    sum = 0;
    ncount = 0;

    for(i=0; i<nval; i++)
    {
        ia = aggindex[i];

        /* Agg index should be increasing */
        if(ia < iaprev)
            return DUTILS_ERROR + __LINE__;

        if(ia != iaprev)
        {
            /* Mean instead of sum */
            if(oper == 1) sum/=ncount;

            outputs[count] = sum;

            count ++;
            if(count >= nval)
                return DUTILS_ERROR + __LINE__;

            sum = 0;
            ncount = 0;
            iaprev = ia;
        }

        sum += inputs[i];
        ncount ++;
    }

    /* Final step */
    if(oper == 1) sum/=ncount;
    outputs[count] = sum;
    iend[0] = count+1;

    return 0;
}


