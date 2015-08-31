import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_crps.h':
    int c_crps(int nval,int ncol,
        int use_weights, int is_sorted,
        double* obs,
        double* sim, 
        double* weights_vector,
        double* reliability_table,
        double* crps_decompos)

cdef extern from 'c_gr4j.h':
    int c_gr4j_getnstates()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnuh()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnoutputs()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getesize()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getuh(double lag,
            int * nuh_optimised,
            double * uh)
  
cdef extern from 'c_gr4j.h':
    int c_gr4j_run(int nval, int nparams, int nuh, int ninputs, 
            int nstates, int noutputs,
    	    double * params,
            double * uh,
    	    double * inputs,
            double * statesuhini,
    	    double * statesini,
            double * outputs)

def gr4j_getnstates():
    return c_gr4j_getnstates()


def gr4j_getnuh():
    return c_gr4j_getnuh()


def gr4j_getnoutputs():
    return c_gr4j_getnoutputs()


def gr4j_getesize():
    return c_gr4j_getesize()


def gr4j_getuh(float lag, 
        np.ndarray[int, ndim=1, mode='c'] nuh_optimised not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None):

    cdef int ierr

    ierr = c_gr4j_getuh(lag,
            <int*> np.PyArray_DATA(nuh_optimised),
            <double*> np.PyArray_DATA(uh))

    return ierr


def gr4j_run(int nuh,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesuhini not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 4
    assert statesini.shape[0] >= 2
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == 2
    assert uh.shape[0] == statesuhini.shape[0]
    assert uh.shape[0] > nuh

    ierr = c_gr4j_run(inputs.shape[0], params.shape[0], nuh, 
            2, 2, outputs.shape[1],
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(uh),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(statesuhini),
            <double*> np.PyArray_DATA(statesini),
            <double*> np.PyArray_DATA(outputs))

    return ierr

def crps(int use_weights, int is_sorted,
        np.ndarray[double, ndim=1, mode='c'] obs not None,
        np.ndarray[double, ndim=2, mode='c'] sim not None,
        np.ndarray[double, ndim=1, mode='c'] weight_vector not None,
        np.ndarray[double, ndim=2, mode='c'] reliability_table not None,
        np.ndarray[double, ndim=1, mode='c'] crps_decompos not None):
    
    cdef int ierr

    # check dimensions
    assert obs.shape[0]==sim.shape[0]
    assert crps_decompos.shape[0]==5
    assert reliability_table.shape[0]==sim.shape[1]+1
    assert reliability_table.shape[1]==7
    
    ierr = c_crps(sim.shape[0], sim.shape[1], 
            use_weights, is_sorted,
            <double*> np.PyArray_DATA(obs),
            <double*> np.PyArray_DATA(sim),
            <double*> np.PyArray_DATA(weight_vector),
            <double*> np.PyArray_DATA(reliability_table),
            <double*> np.PyArray_DATA(crps_decompos))

    return ierr
