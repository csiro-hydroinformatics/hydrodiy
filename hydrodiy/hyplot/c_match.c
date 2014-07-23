#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_combination.h>
#include <gsl/gsl_sf.h>

int c_match(int nval1, int nval2, 
        double * x1, double * y1, 
        double * x2, double * y2, int * match_final){
 
    int i, j, k, * comb_current, * match_current;
    double ndisplay = 1e8, display, count=0., perc, total;
    double distmin = 1e99, dist, dx, dy;

    /* allocate memory */
    comb_current = malloc(sizeof(int)*nval2);
    match_current = malloc(sizeof(int)*nval2);
    gsl_permutation * perm = gsl_permutation_alloc(nval2); 
    gsl_combination * comb = gsl_combination_calloc(nval1, nval2); 

    /* Check inputs */
    if(nval2>nval1) 
        return EDOM;

    /* initialise */
    total = gsl_sf_fact(nval1)/gsl_sf_fact(nval1-nval2);
    printf("\n\tnval1 : %d\n\tnval2 : %d\n\tTotal permut : %0.0f\n",
            nval1, nval2, total);

    /* loop through all permutations of nval2 elements */
    printf("\tComputing distance for all permutations...\n");
    printf("\t... could take a while if nval1 and nval2 are large\n\n");
    printf("\tprogresses :\n");
    do{
        /* Get combination vector */
        //printf("\tcomb current = ");
        for(i=0; i<nval2; i++){
            comb_current[i] = gsl_combination_get(comb, i);
            //printf("%d ", comb_current[i]); 
        }
        //printf("\n");

        gsl_permutation_init(perm); 
        do{
            /* display progresses */
            perc = count/total*100;
            display = fabs(count/ndisplay-floor(count/ndisplay)); 
            if(display < 1e-10) 
                printf("\t\t%3.0f%% (%15.0f)\n", perc, count);

            /* compute total distance */
            dist = 0.;
            for(i=0; i<nval2; i++){
                j = gsl_permutation_get(perm, i);
                k = comb_current[j];
                match_current[i] = k;
                //printf("\t\ti=%d j=%d k=%d\n", i, j ,k);
                dx = x1[k]-x2[i];
                dy = y1[k]-y2[i];
                dist += dx*dx*dx*dx + dy*dy*dy*dy;
            }

            /* save best combination */
            if(dist<distmin){
                distmin = dist;
                for(i=0; i<nval2; i++) 
                    match_final[i] = match_current[i];
            }
            count+=1.;
        }
        while( gsl_permutation_next(perm) == GSL_SUCCESS);
    }
    while( gsl_combination_next(comb) == GSL_SUCCESS);

    printf("\tFinal combination :\n\t\t");
    for(i=0; i<nval2; i++) 
        printf("%d ", match_final[i]);
    printf("\n\tFinal dist : %0.4f\n\n", distmin);

    /* clean up memory */
    free(comb_current);
    free(match_current);
    gsl_permutation_free(perm);
    gsl_combination_free(comb);
    return 0;
}
