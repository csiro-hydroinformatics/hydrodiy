import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_gr4j.h':
    int c_gr4j_getnstates()

cdef extern from 'c_gr4j.h':
    int c_gr4j_getnoutputs()

cdef extern from 'c_gr4j.h':
    int c_gr4j_run(int nval, int nparams, 
            int nuh1, int nuh2,
            int ninputs,
            int nstates, int noutputs,
    	    double * params,
            double * uh1,
            double * uh2,
    	    double * inputs,
            double * statesuh,
    	    double * states,
            double * outputs)

def __cinit__(self):
    pass


def gr4j_getnstates():
    return c_gr4j_getnstates()

def gr4j_getnoutputs():
    return c_gr4j_getnoutputs()

def gr4j_run(int nuh1, 
        int nuh2,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] uh1 not None,
        np.ndarray[double, ndim=1, mode='c'] uh2 not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesuh not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 4:
        raise ValueError('params.shape[0] != 4')
    
    if states.shape[0] < 2:
        raise ValueError('states.shape[0] < 2')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    if uh1.shape[0] < nuh1:
        raise ValueError('uh1.shape[0] < nuh1')

    if uh2.shape[0] < nuh2:
        raise ValueError('uh2.shape[0] < nuh2')

    # Run model
    ierr = c_gr4j_run(inputs.shape[0], 
            params.shape[0], \
            nuh1, 
            nuh2, \
            inputs.shape[1], \
            states.shape[0], \
            outputs.shape[1], \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(uh1), \
            <double*> np.PyArray_DATA(uh2), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesuh), \
            <double*> np.PyArray_DATA(states), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


