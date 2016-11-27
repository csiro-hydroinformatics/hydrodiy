#include "c_dutils.h"

/**
* Aggregate inputs based on the aggindex
* The operator applied to the aggregation is defined by op:
* oper = 0 : sum
* oper = 1 : mean
**/
int c_aggregate(int nval, int oper, int maxnan, int * aggindex,
    double * inputs, double * outputs, int * iend)
{
    int i, cond, nagg, nagg_nan, count, ia, iaprev;
    double agg, inp;

    iaprev = aggindex[0];
    ia = 0;
    count = 0;
    agg = 0;
    nagg = 0;
    nagg_nan = 0;

    for(i=0; i<nval; i++)
    {
        ia = aggindex[i];

        /* Agg index should be increasing */
        if(ia < iaprev)
            return DUTILS_ERROR + __LINE__;

        if(ia != iaprev)
        {
            /* Mean instead of agg */
            if(oper == 1 && nagg>0)
                agg/=nagg;

            /* Store outputs */
            if(nagg_nan > maxnan)
                agg = NAN;

            outputs[count] = agg;

            /* Iterates */
            count ++;
            if(count >= nval)
                return DUTILS_ERROR + __LINE__;

            agg = 0;
            nagg = 0;
            nagg_nan = 0;
            iaprev = ia;
        }

        /* check input and skip value if nan */
        inp = inputs[i];

        if(isnan(inp)){
            nagg_nan ++;
            inp = 0;
        } else
            nagg ++;

        agg += inp;
    }

    /* Final step */
    if(oper == 1 && nagg>0)
        agg/=nagg;

    if(nagg_nan > maxnan)
        agg = NAN;

    outputs[count] = agg;

    iend[0] = count+1;

    return 0;
}


