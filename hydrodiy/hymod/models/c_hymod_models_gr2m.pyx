import numpy as np
cimport numpy as np

np.import_array()

# To force initialisation - Cython compilation bugs otherwise
init = ""

# -- HEADERS --

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

def __cinit__(self):
    pass


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
    if params.shape[0] != 2:
        raise ValueError('params.shape[0] == 2')

    if statesini.shape[0] < 2:
        raise ValueError('statesini.shape[0] >= 2')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] == outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    ierr = c_gr2m_run(inputs.shape[0], \
            params.shape[0], \
            inputs.shape[1], \
            statesini.shape[0], \
            outputs.shape[1], \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesini), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


