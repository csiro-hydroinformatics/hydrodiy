import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --

cdef extern from 'c_crps.h':
    int c_crps(int nval,int ncol,
        int use_weights, int is_sorted,
        double* obs,
        double* sim, 
        double* weights_vector,
        double* reliability_table,
        double* crps_decompos)

cdef extern from 'c_hymod_utils.h':
    int c_hymod_getesize()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnstates()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnuhmax()
 
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnoutputs()
 
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

cdef extern from 'c_gr2m.h':
    int c_gr2m_getnstates()
 
cdef extern from 'c_gr2m.h':
    int c_gr2m_getnoutputs()
 
cdef extern from 'c_gr2m.h':
    int c_gr2m_getesize()
 
cdef extern from 'c_gr2m.h':
    int c_gr2m_run(int nval, int nparams, int ninputs, 
            int nstates, int noutputs,
    	    double * params,
    	    double * inputs,
    	    double * statesini,
            double * outputs)


# -- CRPS --
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


# -- UTILS FUNCTIONS --
def hymod_getesize():
    return c_hymod_getesize()


# -- GR4J FUNCTIONS --

def gr4j_getnstates():
    return c_gr4j_getnstates()

def gr4j_getnuhmax():
    return c_gr4j_getnuhmax()


def gr4j_getnoutputs():
    return c_gr4j_getnoutputs()


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
    assert uh.shape[0] >= nuh

    ierr = c_gr4j_run(inputs.shape[0], params.shape[0], nuh, 
            2, 2, outputs.shape[1],
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(uh),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(statesuhini),
            <double*> np.PyArray_DATA(statesini),
            <double*> np.PyArray_DATA(outputs))

    return ierr


# -- GR2M FUNCTIONS --

def gr2m_getnstates():
    return c_gr2m_getnstates()

def gr2m_getnoutputs():
    return c_gr2m_getnoutputs()

def gr2m_run(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 2
    assert statesini.shape[0] >= 2
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == 2

    ierr = c_gr2m_run(inputs.shape[0], params.shape[0], 
            2, 2, outputs.shape[1],
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(statesini),
            <double*> np.PyArray_DATA(outputs))

    return ierr


