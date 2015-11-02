import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_lagroute.h':
    int c_lagroute_getnstates()

cdef extern from 'c_lagroute.h':
    int c_lagroute_getnoutputs()

cdef extern from 'c_lagroute.h':
    int c_lagroute_run(int nval, 
            int nparams, 
            int nuh,
            int nconfig,
            int ninputs,
            int nstates, 
            int noutputs,
    	    double * config,
    	    double * params,
            double * uh,
    	    double * inputs,
            double * statesuh,
    	    double * states,
            double * outputs)

def __cinit__(self):
    pass


def lagroute_getnstates():
    return c_lagroute_getnstates()

def lagroute_getnoutputs():
    return c_lagroute_getnoutputs()

def lagroute_run(int nuh, 
        np.ndarray[double, ndim=1, mode='c'] config not None,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesuh not None,
        np.ndarray[double, ndim=1, mode='c'] states not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] != 2:
        raise ValueError('params.shape[0] != 2')
    
    if states.shape[0] < 2:
        raise ValueError('states.shape[0] < 2')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 1:
        raise ValueError('inputs.shape[1] != 1')

    if uh.shape[0] < nuh:
        raise ValueError('uh.shape[0] < nuh')

    # Run model
    ierr = c_lagroute_run(inputs.shape[0], 
            params.shape[0], \
            nuh, 
            inputs.shape[1], \
            config.shape[1], \
            states.shape[0], \
            outputs.shape[1], \
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(uh), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesuh), \
            <double*> np.PyArray_DATA(states), \
            <double*> np.PyArray_DATA(outputs))

