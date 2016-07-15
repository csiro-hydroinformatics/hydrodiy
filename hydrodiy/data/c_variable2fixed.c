#include "c_variable2fixed.h"


/*******************************************************************************
 * Routine permettant de convertir une variable a pas de temps variable
 * en variable a pas de temps fixe
 *
 * starthour : hour of the day when the timestep starts
 * shiftday  : Flag to shift the value from end of timestep accumulation to
 *                  beginning
 * maxVTSgap : Maximum lag between two VTS values (in days)
 * timesteplength : Time step duration (in hours)
 * VTSDATE		:	Date of the variable time step values (2d array with each
 *                              date stored as[YYYY, MM, D])
 * VTSSEC		:	Second of the variable time step values (from 0 to 86399)
 * VTSVAL		:	Corresponding value (beware of missing value !!)
 * FTSDATE		:	Date of the variable time step values (2d array with each
 *                              date stored as[YYYY, MM, D])
 * FTSSEC		:	Second of the variable time step values (from 0 to 86399)
 * FTSVAL		:	Fixed values
 *
 *****************************************************************************/
int variable2fixed(int nvalVTS, int nvalFTS,
    int starthour, double maxVTSgap, int timesteplength, int display
    int * VTS_DATE,
    int * VTS_SEC,
    double * VTS_VAL,
    int * FTS_DATE,
    int * FTS_SEC,
    double * FTS_VAL)
{
    int i,VTSindex=0, H, i6, id;
    int QHvaluemiss,Q6Hvaluemiss,QJvaluemiss,nc[1];
    double dates[3];
    double QHvalue, dt1, dt2, ddt1, ddt2, val1, val2;
    double dtH1, dtH2, a, norm, jj, jjprec;

    nc[0]=1;

    // Display status
    if(display==1){
        printf("\tConverting to fixed time step (%d variable ts values -> %d hourly values)..\n",
      		nvalVTS,nvalFTSH);
        printf("\tDaily values computed at %d:00 h\n",starthour);
        printf("\tprogression (percent)..\n\t");
    }

    // Initialisation
    Q6Hvalue=0;
    QJvalue=0;
    i6=0;
    id=0;
    Q6Hvaluemiss=0;

    // Look for the date of the first daily value
    i=0;
    H = FTS_SEC[i]/86400;
    while(H!=starthour){
        i++;
        H = FTS_SEC[i]/86400;
    }
    jj = FTS_DATE[i];
    jjprec=0;

    // Loop through instantaneous data
    for(i=0;i<nvalFTSH-24;i++){

        if(display==1){
            if(i%10000==0)
                printf("%0.1f ",(double)i/(double)nvalFTSH*100);
            if(i%50000==0)
                printf("\n\t");
        }

          DATES[0]=FTSTIMEH[i];c_CONVDATE(nc,FORMAT,DATES,CONV);
        dtH1=CONV[0];
          DATES[0]=FTSTIMEH[i+1];c_CONVDATE(nc,FORMAT,DATES,CONV);

        dtH2=CONV[0];

        while(VTSTIME[VTSindex]<=FTSTIMEH[i])
            VTSindex++;

        VTSindex--;if(VTSindex<0){VTSindex=0;}

        // Initialisation
        QHvalue=0;norm=0;QHvaluemiss=0;QJvaluemiss=0;
        dt2=0;

        // Calculate the hourly value
        //printf("%d %f ~ %f\n",VTSindex,VTSTIME[VTSindex],FTSTIMEH[i+1]);
        while(VTSTIME[VTSindex]<FTSTIMEH[i+1]){

            // Get instants and values of variable time step data
          	DATES[0]=VTSTIME[VTSindex];c_CONVDATE(nc,FORMAT,DATES,CONV);
        	dt1=CONV[0];
          	DATES[0]=VTSTIME[VTSindex+1];c_CONVDATE(nc,FORMAT,DATES,CONV);
        	dt2=CONV[0];
            val1= VTSVAL[VTSindex];
            val2= VTSVAL[VTSindex+1];

        	// prevent the calculation of hourly total
        	// Apply the maximum isolated criteria if both va1 and val2 are strictly positive
            if((val1<0)|(val2<0)|(dt2-dt1>maxVTSgap)){QHvaluemiss=1;}

            // Integrate data
            a=0;if(dt2!=dt1){a = (val2-val1)/(dt2-dt1);}
            ddt1=dt1;if(dt1<dtH1){ddt1=dtH1;}
            ddt2=dt2;if(dt2>dtH2){ddt2=dtH2;}
            QHvalue+=(0.5*a*(ddt1+ddt2-2*dt1)+val1)*(ddt2-ddt1);
            norm+=ddt2-ddt1;
            VTSindex++;
        	//printf("%0.4f %0.2f -> %0.4f %0.2f : %0.4f %0.4f %0.2f\n",
        	//	VTSTIME[VTSindex],val1,VTSTIME[VTSindex+1],val2,FTSTIMEH[i],FTSTIMEH[i+1],QHvalue);
      }

      if(dt2>dtH2)
        VTSindex--;

      if(norm>0 && QHvalue!=-7.7777)
        QHvalue/=norm;
      else
        QHvalue=-7.7777;

      if(QHvaluemiss==1)
        QHvalue=-7.7777;

      // Store Hourly values
      FTSVALH[i]=QHvalue;

      // Store 6Hourly values
      H = (int)(FTSTIMEH[i]*1e-2 - floor(FTSTIMEH[i]*1e-4)*100);
     	//printf("h%d (%f --> %d) v.h=%f v.6h=%f\n",H,FTSTIMEH[i],H%6==0,QHvalue,Q6Hvalue);
     	if(H%6==0){
        FTSVALSIXH[i6]=Q6Hvalue;
        FTSTIMESIXH[i6]=FTSTIMEH[i];
        i6++;
        Q6Hvalue=QHvalue/6;
        if(QHvaluemiss==1){Q6Hvalue=-7.777;Q6Hvaluemiss=1;} // Totals cannot be calculated
        else{Q6Hvaluemiss=0;}
      }
      else{
        if((QHvaluemiss==0) & (Q6Hvaluemiss==0)){Q6Hvalue+=QHvalue/6;}
        else{Q6Hvalue=-7.7777;Q6Hvaluemiss=1;}// Totals cannot be calculated
      }

      // Store daily values
     	//printf("day%0.0f h%d %d QJ=%f\n",jj,H,H==starthour,QJvalue);
      if(H==starthour){
        if(jj!=jjprec){
      	  FTSVALD[id]=QJvalue;
        	  FTSTIMED[id]=jj;
      	  id++;
        	  jjprec=jj;
        }
        QJvalue=QHvalue/24;  // Restart the computation of daily value

        if(QHvaluemiss==1){QJvalue=-7.7777;QJvaluemiss=1;} // Totals cannot be calculated
        else{QJvaluemiss=0;}

        // !!!!!!!!!!!! Daily total from hourStart-Day1 to hourStart-Day2
        // IS AFFECTED TO DAY 2 IF hourStart!=0   !!!!!!!!!!!!!!!!!!!!!!!
        jj=FTSTIMEH[i+24];
        if(endtimestep==0){jj=FTSTIMEH[i];}
      }
      else{
        if((QHvaluemiss==0) & (QJvaluemiss==0)){QJvalue+=QHvalue/24;}
        else{QJvalue=-7.7777;QJvaluemiss=1;}// Totals cannot be calculated
      }

    } // loop on hours
    if(display==1) printf("\n\n");
}

