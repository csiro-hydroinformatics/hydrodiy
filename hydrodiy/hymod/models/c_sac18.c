#include "c_sac18.h"
#include "c_uh.h"

int c_sac18_getnstates(void)
{
    return SAC18_NSTATES;
}


int c_sac18_getnoutputs(void)
{
    return SAC18_NOUTPUTS;
}

int sac18_minmaxparams(int nparams, double * params)
{
    if(nparams<18)
        return ESIZE_PARAMS;

	params[0] = c_utils_minmax(1,1e5,params[0]); 	// S
	params[1] = c_utils_minmax(-50,50,params[1]);	// IGF
	params[2] = c_utils_minmax(1,1e5,params[2]); 	// R
	params[3] = c_utils_minmax(0.5,50,params[3]); // TB

	return 0;
}



int sac18_runtimestep(int nparams,
    int nuh, 
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh,
    double * inputs,
    double * statesuh,
    double * states,
    double * outputs)
{
    int k,ii;
    int itime, ninc,inc; //! increment limits or counters
    double Pet=0;
    double rainfall=0;
    double pliq=0; // redundant with rainfall, but keep the variable name from the original code
    double evapt=0; //  ! was evap / evapt in original subroutines

    // Sac parameters
    double Adimp,Lzfpm,Lzfsm,Lzpk,Lzsk,Lztwm,Pctim,Pfree,Rexp,Rserv;
    double Sarva,Side,Ssout,Uzfwm,Uzk,Uztwm,Zperc;

    // The following constants were used with no explanation in the 
    // original Fortran code...
    double PDN20 = 5.08;
    double PDNOR = 25.4;

    // States
    double Uztwc=states[0];
    double Uzfwc=states[1];
    double Lztwc=states[2];
    double Lzfsc=states[3];
    double Lzfpc=states[4];
    double Adimc=states[5];

    double Alzfpc=0;
    double Alzfpm=0;
    double Alzfsc=0;
    double Alzfsm=0;
    double ChannelFlow=0;
    double E3=0;
    double E5=0;
    double EvaporationChannelWater=0;
    double EvapUzfw=0;
    double EvapUztw=0;
    double Flobf=0;
    double Floin=0;
    double Flosf=0;
    double Flwbf=0;
    double Flwsf=0;
    //double Lzmpd=0;
    double Pbase=0;
    double Perc=0;
    double ReservedLowerZone=0;
    double Roimp=0;
    //double SumLowerZoneCapacities=0;
    double totalBeforeChannelLosses=0;
    double ratioBaseflow=0;
    double runoff=0;
    double baseflow=0;
    double HydrographStore=0;

    // For reference and traceability: some variables named a and b (sic) in the original code 
    // were used in alternance for different purposes, instead of the following four following variables
    double ratioUztw=0, ratioUzfw=0;
    double ratioLztw=0, ratioLzfw=0;
    double transfered=0; //! was named del in DLWC code

    double addro=0, adj=0, bf=0, dinc=0, dlzp=0, dlzs=0;
    double duz=0, hpl=0, lzair=0, pav=0, percfw=0, percs=0;
    double perctw=0, pinc=0, ratio=0, ratlp=0, ratls=0;
    double x;
   
    // Affect parameters after checking bounds
    Adimp = params[0];
    Lzfpm = params[1];
    Lzfsm = params[2];
    Lzpk  = params[3];
    Lzsk  = params[4];
    Lztwm = params[5];
    Pctim = params[6];
    Pfree = params[7];
    Rexp  = params[8];
    Rserv = params[9];
    Sarva = params[10];
    Side  = params[11];
    Ssout = params[12];
    Uzfwm = params[14];
    Uzk   = params[15];
    Uztwm = params[16];
    Zperc = params[17];

    ReservedLowerZone = Rserv * ( Lzfpm + Lzfsm );	
    Alzfsm = Lzfsm * ( 1.0 + Side );
    Alzfpm = Lzfpm * ( 1.0 + Side );
    Alzfsc = Lzfsc * ( 1.0 + Side );
    Alzfpc = Lzfpc * ( 1.0 + Side );

    Pbase = Alzfsm * Lzsk + Alzfpm * Lzpk;

    // inputs
    rainfall = inputs[0];
    rainfall= rainfall < 0. ? 0. : rainfall;
    
    Pet = inputs[1];
    Pet = Pet < 0. ? 0. : Pet;
   
    // At this point in the Fortran implementation, there were some pan factors applied. 
    // This is not included here. A modified time series should be fed into the PET.
    evapt = Pet;
    pliq = rainfall;
    
    //! Determine evaporation from upper zone tension water store
    if( Uztwm > 0.0 )
        EvapUztw = evapt * Uztwc / Uztwm;
    else
        EvapUztw = 0.0;
    
    //! Determine evaporation from upper zone free water
    if( Uztwc < EvapUztw )
    {
        EvapUztw = Uztwc;
        Uztwc = 0.0;
        //!     Determine evaporation from free water surface
        EvapUzfw = min( ( evapt - EvapUztw ), Uzfwc );
        Uzfwc = Uzfwc - EvapUzfw;
    }
    else
    {
        Uztwc = Uztwc - EvapUztw;
        EvapUzfw = 0.0;
    }
    
    //     If the upper zone free water ratio exceeded the upper tension zone
    //     content ratio, then transfer the free water into tension until the ratios are equals
    if( Uztwm > 0.0 )
        ratioUztw = Uztwc / Uztwm;
    else
        ratioUztw = 1.0;
    
    if( Uzfwm > 0.0 )
        ratioUzfw = Uzfwc / Uzfwm;
    else
        ratioUzfw = 1.0;
    
    if( ratioUztw < ratioUzfw )
    {
        //! equivalent to the tension zone "sucking" the free water
        ratioUztw = ( Uztwc + Uzfwc ) / ( Uztwm + Uzfwm );
        Uztwc = Uztwm * ratioUztw;
        Uzfwc = Uzfwm * ratioUztw;
    }

    //     Evaporation from Adimp (additional impervious area) and Lower zone tension water
    if( Uztwm + Lztwm > 0.0 )
    {
    	x =  ( evapt - EvapUztw - EvapUzfw ) * Lztwc / ( Uztwm + Lztwm );
        E3 = min(x, Lztwc );
    
    	x = EvapUztw + ( ( evapt - EvapUztw - EvapUzfw ) * ( Adimc - EvapUztw - Uztwc ) / ( Uztwm + Lztwm ) );
        E5 = min(x, Adimc );
    }
    else
    {
        E3 = 0.0;
        E5 = 0.0;
    }
    
    //     Compute the *transpiration*  loss from the lower zone tension
    Lztwc = Lztwc - E3;
    
    //     Adjust the impervious area store
    Adimc = Adimc - E5;
    EvapUztw = EvapUztw * ( 1 - Adimp - Pctim );
    EvapUzfw = EvapUzfw * ( 1 - Adimp - Pctim );
    E3 = E3 * ( 1 - Adimp - Pctim );
    E5 = E5 * Adimp;
    
    //     Resupply the lower zone tension with water from the lower zone
    //     free water, if more water is available there.
    if( Lztwm > 0.0 )
        ratioLztw = Lztwc / Lztwm;
    else
        ratioLztw = 1.0;
    
    if( Alzfpm + Alzfsm - ReservedLowerZone + Lztwm > 0.0 )
        ratioLzfw = ( Alzfpc + Alzfsc - ReservedLowerZone + Lztwc ) /
                    ( Alzfpm + Alzfsm - ReservedLowerZone + Lztwm );
    else
        ratioLzfw = 1.0;
    
    if( ratioLztw < ratioLzfw )
    {
        transfered = ( ratioLzfw - ratioLztw ) * Lztwm;
        //       Transfer water from the lower zone secondary free water to lower zone
        //       tension water store
        Lztwc = Lztwc + transfered;
        Alzfsc = Alzfsc - transfered;
        if( Alzfsc < 0 )
        {
            //         Transfer primary free water if secondary free water is inadequate
            Alzfpc = Alzfpc + Alzfsc;
            Alzfsc = 0.0;
        }
    }
    
    //     Runoff from the impervious or water covered area
    Roimp = pliq * Pctim;
    
    //     Reduce the rain by the amount of upper zone tension water deficiency
    pav = pliq + Uztwc - Uztwm;
    if( pav < 0 )
    {
        //!Fill the upper zone tension water as much as rain permits
        Adimc = Adimc + pliq;
        Uztwc = Uztwc + pliq;
        pav = 0.0;
    }
    else
    {
        Adimc = Adimc + Uztwm - Uztwc;
        Uztwc = Uztwm;
    }
    
    // The rest of this method is very close to the original Fortran implementation; 
    // Given the look of it I doubt I can get things to reproduce from first principle.
    if( pav <= PDN20 )
    {
        adj = 1.0;
        itime = 2;
    }
    else
    {
        if( pav < PDNOR )
        {
            //! Effective rainfall in a period is assumed to be half of the
            //! period length for rain equal to the normal rainy period
            adj = 0.5 * pow( pav / PDNOR ,0.5);
        }
        else
        {
            adj = 1.0 - 0.5 * PDNOR / pav;
        }
        itime = 1;
    }
    
    Flobf = 0.0;
    Flosf = 0.0;
    Floin = 0.0;
    
    //! Here again, being blindly faithful to original implementation
    hpl = Alzfpm / ( Alzfpm + Alzfsm );
    
    for(ii = itime; ii <= 2; ii++ )
    {
        // using (int) Math.Floor to reproduce the fortran INT cast, even if I think (int) would do.
        ninc = (int)( ( Uzfwc * adj + pav ) * 0.2 ) + 1;
        dinc = 1.0 / ninc;
        pinc = pav * dinc;
        dinc = dinc * adj;
        if( ninc == 1 && adj >= 1.0 )
        {
            duz = Uzk;
            dlzp = Lzpk;
            dlzs = Lzsk;
        }
        else
        {
            if( Uzk < 1.0 )
            {
                duz = 1.0 - pow( ( 1.0 - Uzk ), dinc );
            }
            else
                duz = 1.0;
    
            if( Lzpk < 1.0 )
                dlzp = 1.0 - pow( ( 1.0 - Lzpk ), dinc );
            else
                dlzp = 1.0;
    
            if( Lzsk < 1.0 )
                dlzs = 1.0 - pow( ( 1.0 - Lzsk ), dinc );
            else
                dlzs = 1.0;
        }
    
        //       Drainage and percolation loop
        for(inc = 1; inc <= ninc; inc++ )
        {
            ratio = ( Adimc - Uztwc ) / Lztwm;
            addro = pinc * ratio * ratio;
    
            //         Compute the baseflow from the lower zone
            if( Alzfpc > 0.0 )
                bf = Alzfpc * dlzp;
            else
            {
                Alzfpc = 0.0;
                bf = 0.0;
    
            }
    
            Flobf = Flobf + bf;
            Alzfpc = Alzfpc - bf;
    
            if( Alzfsc > 0.0 )
                bf = Alzfsc * dlzs;
            else
            {
    
                Alzfsc = 0.0;
                bf = 0.0;
            }
    
            Alzfsc = Alzfsc - bf;
            Flobf = Flobf + bf;
    
            //         Adjust the upper zone for percolation and interflow
            if( Uzfwc > 0.0 )
            {
                //           Determine percolation from the upper zone free water
                //	limited to available water and lower zone air space
                lzair = Lztwm - Lztwc + Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                if( lzair > 0.0 )
                {
                    Perc = ( Pbase * dinc * Uzfwc ) / Uzfwm;
    
    				x = Perc * ( 1.0 +( Zperc * pow(( 1.0 
    						- ( Alzfpc + Alzfsc + Lztwc ) / ( Alzfpm + Alzfsm + Lztwm ) ),Rexp ) ) );
                    Perc = min( Uzfwc,x);
                    Perc = min( lzair, Perc );
                    Uzfwc = Uzfwc - Perc;
    
    				if(debugflag>2){ 
    					printf("\t\t\tLoop ii=%d - inc=%d, Pbase=%f dinc=%f Uzfwc=%f Uzfwm=%f lzair=%f\n",
    									ii,inc,Pbase,dinc,Uzfwc,Uzfwm,lzair);
    				}
    
                }
                else
                    Perc = 0.0;
    
                //           Compute the interflow
                transfered = duz * Uzfwc;
                Floin = Floin + transfered;
                Uzfwc = Uzfwc - transfered;
    
                //           Distribute water to lower zone tension and free water stores
                perctw = min( Perc * ( 1.0 - Pfree ), Lztwm - Lztwc );
                percfw = Perc - perctw;
                //           Shift any excess lower zone free water percolation to the
                //           lower zone tension water store
                lzair = Alzfsm - Alzfsc + Alzfpm - Alzfpc;
                if( percfw > lzair )
                {
                    perctw = perctw + percfw - lzair;
                    percfw = lzair;
                }
                Lztwc = Lztwc + perctw;
    
                //           Distribute water between LZ free water supplemental and primary
                if( percfw > 0.0 )
                {
                    ratlp = 1.0 - Alzfpc / Alzfpm;
                    ratls = 1.0 - Alzfsc / Alzfsm;
                    percs = min( Alzfsm - Alzfsc,
                                      percfw * ( 1.0 - hpl * ( 2.0 * ratlp ) / ( ratlp + ratls ) ) );
                    Alzfsc = Alzfsc + percs;
    
                    //             Check for spill from supplemental to primary
                    if( Alzfsc > Alzfsm )
                    {
                        percs = percs - Alzfsc + Alzfsm;
                        Alzfsc = Alzfsm;
                    }
                    Alzfpc = Alzfpc + percfw - percs;
    
                    //             Check for spill from primary to supplemental
                    if( Alzfpc > Alzfpm )
                    {
                        Alzfsc = Alzfsc + Alzfpc - Alzfpm;
                        Alzfpc = Alzfpm;
                    }
                }
            }
    
            //         Fill upper zone free water with tension water spill
            if( pinc > 0.0 )
            {
                pav = pinc;
                if( pav - Uzfwm + Uzfwc <= 0 )
                    Uzfwc = Uzfwc + pav;
                else
                {
                    pav = pav - Uzfwm + Uzfwc;
                    Uzfwc = Uzfwm;
                    Flosf = Flosf + pav;
                    addro = addro + pav * ( 1.0 - addro / pinc );
                }
            }
            Adimc = Adimc + pinc - addro;
            Roimp = Roimp + addro * Adimp;
        }
        adj = 1.0 - adj;
        pav = 0.0;
    }
    //     Compute the storage volumes, runoff components and evaporation
    //     Note evapotranspiration losses from the water surface and
    //     riparian vegetation areas are computed in stn7a
    Flosf = Flosf * ( 1.0 - Pctim - Adimp );
    Floin = Floin * ( 1.0 - Pctim - Adimp );
    Flobf = Flobf * ( 1.0 - Pctim - Adimp );
    
    //  !!!!!!!!!!!!!! end of call to stn7b 
    //  !!!!! following code to the end of the subroutine is part of stn7a
    
    Lzfsc = Alzfsc / ( 1.0 + Side );
    Lzfpc = Alzfpc / ( 1.0 + Side );
    
    /// Perform the routing of the surface runoff 
    /// via UH
    for (k=0;k<*nuh-1;k++) statesuh[k]=statesuh[1+k]+uh[k]*(Flosf + Roimp + Floin);
    statesuh[*nuh-1]=uh[*nuh-1]*(Flosf + Roimp + Floin);
    Flwsf = statesuh[0];
    HydrographStore += ( Flosf + Roimp + Floin - Flwsf);
    
    Flwbf = Flobf / ( 1.0 + Side );
    if( Flwbf < 0.0 )
        Flwbf = 0.0;
    
    // Calculate the BFI prior to losses, in order to keep 
    // this ratio in the final runoff and baseflow components.
    totalBeforeChannelLosses = Flwbf + Flwsf;
    ratioBaseflow = 0;
    if( totalBeforeChannelLosses > 0 )
        ratioBaseflow = Flwbf / totalBeforeChannelLosses;
    
    //! Subtract losses from the total channel flow ( going to the subsurface discharge )
    ChannelFlow = max( 0.0, ( Flwbf + Flwsf - Ssout ) );
    
    //! following was e4 
    EvaporationChannelWater = min( evapt * Sarva, ChannelFlow );
    
    runoff = ChannelFlow - EvaporationChannelWater;
    baseflow = runoff * ratioBaseflow;
    
    /*--- RESULTS ---------------- */
    states[0]   =   Uztwc;
    states[1]   =   Uzfwc;
    states[2]   =   Lztwc;
    states[3]   =   Lzfsc;
    states[4]   =   Lzfpc;
    states[5]   =   Adimc;

    outputs[0] = runoff;

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

    return ierr;
}


/* --------- Model runner ----------*/
int c_sac18_run(int nval, 
    int nparams,
    int nuh, 
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh,
    double * inputs,
    double * statesuh,
    double * states,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(noutputs > SAC18_NOUTPUTS)
        return ESIZE_OUTPUTS;

    if(nuh > NUHMAXLENGTH)
        return ESIZE_STATESUH;

    if(nuh <= 0)
        return ESIZE_STATESUH;

    /* Check parameters */
    ierr = sac18_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = 0; i < nval; i++)
    {
        /* Run timestep model and update states */
    	ierr = sac18_runtimestep(nparams,
                nuh,
                ninputs,
                nstates,
                noutputs,
                params,
                uh,
                &(inputs[ninputs*i]),
                statesuh,
                states,
                &(outputs[noutputs*i]));

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

